from __future__ import annotations

"""
model_registry.py

Centralized registry and loader for all ML/NLP models used in the project.
Supports SentenceTransformer, spaCy, and custom models.
Ensures models are loaded once, cached, and reused across modules.
Integrates with config.py for model directory paths.
Optimized for robust, singleton-style loading, device selection, path validation, and logging.
"""
import os
import re
import subprocess
import sys
import threading
from collections import Counter
from typing import Any, Callable, Dict

# Lazy/defensive torch import: avoid hard failure on environments where
# DLLs are unavailable. Downstream code must check availability.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]
from selectolax.parser import HTMLParser

from ..config import MODEL_DIR, PROJECT_ROOT, TABLE_MODEL_PATH, VOCAB_DIR
from ..Context_Integration.librarian import load_context_library
from .logger_singleton import logger
SentenceTransformer = None  # defer import to use-sites
spacy = None  # defer import to use-sites
Language = None
_lock = threading.Lock()

def _hf_offline() -> bool:
    return (
        os.getenv("TRANSFORMERS_OFFLINE") == "1"
        or os.getenv("HUGGINGFACE_HUB_OFFLINE") == "1"
        or os.getenv("DISABLE_HF_DOWNLOAD") == "1"
    )

# --- Robust Vocabulary Loading Utilities ---

def load_vocab_from_file(path: str) -> Dict[str, int]:
    """
    Loads a vocabulary file where each line is a token/label.
    Returns a dict mapping token -> index (starting from 1).
    """
    vocab = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                word = line.strip()
                if word:
                    if word in vocab:
                        logger.error(f"Duplicate token '{word}' in vocab file {path}")
                    vocab[word] = idx + 1  # 0 reserved for padding/OOV
    else:
        logger.error(f"Vocab file not found: {path}")
    return vocab

def build_reverse_vocab(vocab: Dict[str, int], cast_int: bool = False) -> Dict[int, Any]:
    """
    Builds a reverse mapping from index to token.
    Optionally casts keys to int (for years).
    """
    if cast_int:
        return {v: int(k) for k, v in vocab.items()}
    return {v: k for k, v in vocab.items()}

# --- Dynamic Vocabulary Initialization ---

WORD2IDX = load_vocab_from_file(os.path.join(VOCAB_DIR, "words.txt"))
STATE2IDX = load_vocab_from_file(os.path.join(VOCAB_DIR, "states.txt"))
COUNTY2IDX = load_vocab_from_file(os.path.join(VOCAB_DIR, "counties.txt"))
TYPE2IDX = load_vocab_from_file(os.path.join(VOCAB_DIR, "types.txt"))
YEAR2IDX = load_vocab_from_file(os.path.join(VOCAB_DIR, "years.txt"))

IDX2STATE = build_reverse_vocab(STATE2IDX)
IDX2COUNTY = build_reverse_vocab(COUNTY2IDX)
IDX2TYPE = build_reverse_vocab(TYPE2IDX)
IDX2YEAR = build_reverse_vocab(YEAR2IDX, cast_int=True)

# --- Advanced Tokenizer ---

def advanced_tokenizer(text: str, max_len: int = 20):
    """
    Tokenizes text using the project vocabulary.
    Returns a torch.Tensor when torch is available; otherwise raises ImportError.
    """
    tokens = re.findall(r"\w+", text.lower())
    idxs = [WORD2IDX.get(tok, 0) for tok in tokens]
    idxs = idxs[:max_len] + [0] * (max_len - len(idxs))
    if torch is None:
        raise ImportError("Torch is not available for advanced_tokenizer")
    return torch.tensor(idxs).unsqueeze(0)  # (1, max_len)

# --- ContestFieldClassifier (Multi-head, Multi-field) ---

if torch is not None and nn is not None and F is not None:
    class ContestFieldClassifier(nn.Module):
        """
        Predicts contest fields (year, state, county, type_) from contest title text.
        Returns predictions with confidence and explanations.
        """
        def __init__(self, vocab_size, embed_dim, num_years, num_states, num_counties, num_types):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
            self.year_head = nn.Linear(embed_dim * 2, num_years)
            self.state_head = nn.Linear(embed_dim * 2, num_states)
            self.county_head = nn.Linear(embed_dim * 2, num_counties)
            self.type_head = nn.Linear(embed_dim * 2, num_types)

        def forward(self, x):
            emb = self.embedding(x)
            _, (h_n, _) = self.encoder(emb)
            h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, embed_dim*2)
            year_logits = self.year_head(h)
            state_logits = self.state_head(h)
            county_logits = self.county_head(h)
            type_logits = self.type_head(h)
            return year_logits, state_logits, county_logits, type_logits

        @classmethod
        def load_from_checkpoint(cls, path) -> "ContestFieldClassifier":
            # Load vocab sizes dynamically
            model = cls(
                vocab_size=max(WORD2IDX.values(), default=1) + 1,
                embed_dim=128,
                num_years=max(YEAR2IDX.values(), default=1) + 1,
                num_states=max(STATE2IDX.values(), default=1) + 1,
                num_counties=max(COUNTY2IDX.values(), default=1) + 1,
                num_types=max(TYPE2IDX.values(), default=1) + 1
            )
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model

        def predict(self, text: str) -> Dict[str, Dict[str, Any]]:
            """
            Predict all contest fields from text.
            Returns a dict: {field: {"value": ..., "confidence": ..., "explanation": ...}}
            """
            x = advanced_tokenizer(text)
            with torch.no_grad():
                year_logits, state_logits, county_logits, type_logits = self.forward(x)
                year_probs = F.softmax(year_logits, dim=1).squeeze()
                state_probs = F.softmax(state_logits, dim=1).squeeze()
                county_probs = F.softmax(county_logits, dim=1).squeeze()
                type_probs = F.softmax(type_logits, dim=1).squeeze()

                year_idx = int(year_probs.argmax())
                state_idx = int(state_probs.argmax())
                county_idx = int(county_probs.argmax())
                type_idx = int(type_probs.argmax())

                result = {
                    "year": {
                        "value": IDX2YEAR.get(year_idx, None),
                        "confidence": float(year_probs[year_idx]),
                        "explanation": f"Predicted year from text, top prob: {float(year_probs[year_idx]):.2f}"
                    },
                    "state": {
                        "value": IDX2STATE.get(state_idx, None),
                        "confidence": float(state_probs[state_idx]),
                        "explanation": f"Predicted state from text, top prob: {float(state_probs[state_idx]):.2f}"
                    },
                    "county": {
                        "value": IDX2COUNTY.get(county_idx, None),
                        "confidence": float(county_probs[county_idx]),
                        "explanation": f"Predicted county from text, top prob: {float(county_probs[county_idx]):.2f}"
                    },
                    "type_": {
                        "value": IDX2TYPE.get(type_idx, None),
                        "confidence": float(type_probs[type_idx]),
                        "explanation": f"Predicted type from text, top prob: {float(type_probs[type_idx]):.2f}"
                    }
                }
                return result
else:
    class ContestFieldClassifier:  # type: ignore[no-redef]
        @classmethod
        def load_from_checkpoint(cls, path):
            raise ImportError("Torch is not available: cannot load ContestFieldClassifier")

        def predict(self, text: str):
            raise ImportError("Torch is not available: predict() unsupported")

if torch is not None and nn is not None and F is not None:
    class CandidateClassifier(nn.Module):
        """
        Predicts candidate from input text.
        Returns predictions with confidence and explanations.
        """
        def __init__(self, vocab_size, embed_dim, num_candidates):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.encoder = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
            self.candidate_head = nn.Linear(embed_dim * 2, num_candidates)

        def forward(self, x):
            emb = self.embedding(x)
            _, (h_n, _) = self.encoder(emb)
            h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, embed_dim*2)
            candidate_logits = self.candidate_head(h)
            return candidate_logits

        @classmethod
        def load_from_checkpoint(cls, path, vocab_size, embed_dim, num_candidates):
            model = cls(vocab_size, embed_dim, num_candidates)
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model

        def predict(self, text, tokenizer, idx2candidate):
            """
            Predict candidate from text.
            Args:
                text: str, input text
                tokenizer: function, returns torch tensor of token indices
                idx2candidate: dict, maps index to candidate string
            Returns:
                dict: {"value": ..., "confidence": ..., "explanation": ...}
            """
            x = tokenizer(text)
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1).squeeze()
                idx = int(probs.argmax())
                value = idx2candidate.get(idx, None)
                confidence = float(probs[idx])
                explanation = f"Predicted candidate from text, top prob: {confidence:.2f}"
                return {
                    "value": value,
                    "confidence": confidence,
                    "explanation": explanation
                }
else:
    class CandidateClassifier:  # type: ignore[no-redef]
        @classmethod
        def load_from_checkpoint(cls, path, vocab_size, embed_dim, num_candidates):
            raise ImportError("Torch is not available: cannot load CandidateClassifier")

        def predict(self, text, tokenizer, idx2candidate):
            raise ImportError("Torch is not available: predict() unsupported")
# --- ModelRegistry Integration ---

class ModelRegistry(object):
    """
    Centralized registry for ML/NLP models.
    Supports:
      - Fine-tuned and base SentenceTransformer models
      - Fine-tuned and base spaCy models
      - Arbitrary custom models (e.g., sklearn, torch, etc.)
    Optimized for singleton-style loading, device selection, and robust error handling.
    """
    _models = {}
    _nlp_models = {}
    _custom_models = {}
    _model_paths = {
        "sentence_transformer": os.path.join(MODEL_DIR, "fine_tuned_table_headers"),
        "spacy_ner": os.path.join(MODEL_DIR, "fine_tuned_spacy_ner"),
        "torch_contest": os.path.join(MODEL_DIR, "contest_field_classifier.pt"),
        # Add more as needed
    }
    _loaded_info = {}
    _spacy_model = None
    _torch_contest_model = None

    @classmethod
    def get_spacy_model(cls, model_name=None, use_finetuned=True):
        # Lazy import of spaCy to avoid thinc->torch import chain at module import
        global spacy
        if spacy is None:
            try:
                import spacy as _spacy
                spacy = _spacy
            except Exception as e:
                raise ImportError(f"spaCy unavailable: {e}")
        with _lock:
            key = f"spacy:{model_name or 'default'}:{use_finetuned}"
            if key in cls._nlp_models:
                return cls._nlp_models[key]
            # Try fine-tuned model first
            if use_finetuned:
                finetuned_path = cls._model_paths["spacy_ner"]
                meta_path = os.path.join(finetuned_path, "meta.json")
                if os.path.exists(meta_path):
                    logger.info(f"Loading fine-tuned spaCy model from {finetuned_path}")
                    try:
                        nlp = spacy.load(finetuned_path)
                        cls._nlp_models[key] = nlp
                        cls._loaded_info[key] = finetuned_path
                        return nlp
                    except Exception as e:
                        logger.error(f"Failed to load fine-tuned spaCy model: {e}")
            # Fallback to base model
            base_name = model_name or "en_core_web_sm"
            try:
                logger.info(f"Loading base spaCy model: {base_name}")
                nlp = spacy.load(base_name)
                cls._nlp_models[key] = nlp
                cls._loaded_info[key] = base_name
                return nlp
            except OSError:
                # Auto-download if missing
                logger.info(f"Downloading spaCy model: {base_name}")
                subprocess.run([sys.executable, "-m", "spacy", "download", base_name], check=True, cwd=PROJECT_ROOT)
                nlp = spacy.load(base_name)
                cls._nlp_models[key] = nlp
                cls._loaded_info[key] = base_name
                return nlp
            except Exception as e:
                logger.error(f"Failed to load base spaCy model: {e}")
                raise

    @classmethod
    def get_torch_contest_model(cls) -> "ContestFieldClassifier":
        if torch is None:
            raise ImportError("Torch is not available in this environment.")
        if cls._torch_contest_model is None:
            model_path = cls._model_paths["torch_contest"]
            if not os.path.exists(model_path):
                logger.error(f"Torch contest model checkpoint not found: {model_path}")
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            cls._torch_contest_model = ContestFieldClassifier.load_from_checkpoint(model_path)
            cls._loaded_info["torch_contest_model"] = model_path
        return cls._torch_contest_model

    # Example: Add more models for other DB objects (candidates, offices, etc.)
    _torch_candidate_model = None

    @classmethod
    def get_torch_candidate_model(cls) -> "CandidateClassifier":
        """
        Loads and caches the torch-based CandidateClassifier model.
        Dynamically builds vocab from librarian.py if available.
        """
        if torch is None:
            logger.error("Torch is not available in this environment.")
            return None
        if cls._torch_candidate_model is not None:
            return cls._torch_candidate_model

        # --- Dynamic vocab loading from librarian.py ---
        try:
            context = load_context_library()
            candidate_vocab = context.get("candidate_keywords", [])
            CANDIDATE2IDX = {c: i + 1 for i, c in enumerate(sorted(set(candidate_vocab)))}
        except Exception as e:
            logger.error(f"Failed to load candidate vocab from librarian.py: {e}")
            CANDIDATE2IDX = {}

        # --- Model loading ---
        model_path = cls._model_paths.get("torch_candidate", os.path.join(MODEL_DIR, "candidate_classifier.pt"))
        if not os.path.exists(model_path):
            logger.error(f"Torch candidate model checkpoint not found: {model_path}")
            return None

        try:
            logger.info(f"Loading CandidateClassifier from {model_path}")
            vocab_size = len(CANDIDATE2IDX) + 1
            model = CandidateClassifier.load_from_checkpoint(
                model_path,
                vocab_size=vocab_size,
                embed_dim=128,
                num_candidates=vocab_size
            )
            model.eval()
            cls._torch_candidate_model = model
            cls._loaded_info["torch_candidate_model"] = model_path
            return model
        except Exception as e:
            logger.error(f"Failed to load CandidateClassifier: {e}")
            return None

    @classmethod
    def get_loaded_models_info(cls) -> dict:
        return dict(cls._loaded_info)

    @classmethod
    def _get_device(cls) -> str:
        if torch is not None and torch.cuda.is_available():
            logger.info("Using CUDA for model loading.")
            return "cuda"
        logger.info("Using CPU for model loading.")
        return "cpu"

    @classmethod
    def get_sentence_transformer(cls, model_name=None, use_finetuned=True, device=None):
        # Lazy import to avoid heavy dependencies during module import
        global SentenceTransformer
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST
                SentenceTransformer = _ST
            except Exception as e:
                raise ImportError(f"sentence_transformers unavailable: {e}")
        with _lock:
            base_name = model_name or "all-MiniLM-L6-v2"
            if not isinstance(base_name, str) or not base_name.strip() or base_name.startswith("SentenceTransformer("):
                logger.error(f"Invalid model_name for SentenceTransformer: {base_name!r}. Using default 'all-MiniLM-L6-v2'.")
                base_name = "all-MiniLM-L6-v2"
            key = f"sentence_transformer:{base_name}:{use_finetuned}"
            if key in cls._models:
                return cls._models[key]

            # 0) Optional explicit local path override
            local_override = os.getenv("SENTENCE_TRANSFORMER_LOCAL_PATH")  # e.g., C:\models\all-MiniLM-L6-v2
            if local_override and os.path.isdir(local_override):
                try:
                    logger.info(f"Loading SentenceTransformer from local override: {local_override}")
                    model = SentenceTransformer(local_override, device=device or cls._get_device())
                    cls._models[key] = model
                    cls._loaded_info[key] = local_override
                    return model
                except Exception as e:
                    logger.warning(f"Failed loading local override for SentenceTransformer: {e}")

            # 1) Try fine-tuned local directory
            if use_finetuned:
                finetuned_path = cls._model_paths["sentence_transformer"]
                config_path = os.path.join(finetuned_path, "config.json")
                if os.path.exists(config_path):
                    logger.info(f"Loading fine-tuned SentenceTransformer from {finetuned_path}")
                    try:
                        model = SentenceTransformer(finetuned_path, device=device or cls._get_device())
                        cls._models[key] = model
                        cls._loaded_info[key] = finetuned_path
                        return model
                    except Exception as e:
                        logger.error(f"Failed to load fine-tuned SentenceTransformer: {e}")
                        cls._models[key] = None
                        return None

            # 2) If offline, do not attempt network download
            if _hf_offline():
                logger.warning("TRANSFORMERS_OFFLINE/HUGGINGFACE_HUB_OFFLINE set; skipping HF download. Embeddings disabled.")
                cls._models[key] = None
                return None

            # 3) Try base model (may hit network)
            logger.info(f"Loading base SentenceTransformer: {base_name}")
            try:
                # Optional: honor HF cache dirs if present
                cache_folder = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or None
                kwargs = {"device": device or cls._get_device()}
                if cache_folder:
                    kwargs["cache_folder"] = cache_folder
                model = SentenceTransformer(base_name, **kwargs)
                cls._models[key] = model
                cls._loaded_info[key] = base_name
                return model
            except Exception as e:
                # Downgrade DNS/network errors to WARNING for noisy environments
                msg = str(e)
                if "NameResolutionError" in msg or "MaxRetryError" in msg or "Failed to resolve" in msg:
                    logger.warning(f"Failed to load base SentenceTransformer (network/DNS). Running without embeddings. Error: {e}")
                else:
                    logger.error(f"Failed to load base SentenceTransformer: {e}")
                cls._models[key] = None
                return None

    @classmethod
    def get_custom_model(cls, key: str, loader_func: Callable, *args, **kwargs) -> Any:
        with _lock:
            if key not in cls._custom_models:
                logger.info(f"Loading custom model: {key}")
                try:
                    cls._custom_models[key] = loader_func(*args, **kwargs)
                    cls._loaded_info[key] = str(loader_func)
                except Exception as e:
                    logger.error(f"Failed to load custom model {key}: {e}")
                    raise
            return cls._custom_models[key]

    @classmethod
    def reload_model(cls, model_type, model_name=None) -> None:
        """
        Reloads the specified model type and name.
        """
        with _lock:
            if model_type == "sentence_transformer":
                key_prefix = f"sentence_transformer:{model_name or 'default'}"
                cls._models = {k: v for k, v in cls._models.items() if not k.startswith(key_prefix)}
                cls._loaded_info = {k: v for k, v in cls._loaded_info.items() if not k.startswith(key_prefix)}
            elif model_type == "spacy":
                key_prefix = f"spacy:{model_name or 'default'}"
                cls._nlp_models = {k: v for k, v in cls._nlp_models.items() if not k.startswith(key_prefix)}
                cls._loaded_info = {k: v for k, v in cls._loaded_info.items() if not k.startswith(key_prefix)}
            else:
                if model_type in cls._custom_models:
                    del cls._custom_models[model_type]
                if model_type in cls._loaded_info:
                    del cls._loaded_info[model_type]
            logger.info(f"Reloaded model(s) of type: {model_type}")

    @classmethod
    def clear_cache(cls) -> None:
        with _lock:
            cls._models.clear()
            cls._nlp_models.clear()
            cls._custom_models.clear()
            cls._loaded_info.clear()
            logger.info("Model registry cache cleared.")

    @classmethod
    def set_model_path(cls, model_type, path) -> None:
        cls._model_paths[model_type] = path

    @classmethod
    def get_model_name(cls, model) -> str:
        if hasattr(model, 'model_name_or_path'):
            return getattr(model, 'model_name_or_path')
        if hasattr(model, 'modules') and hasattr(model.modules[0], 'model_name_or_path'):
            return getattr(model.modules[0], 'model_name_or_path')
        return str(model)

if torch is not None and nn is not None and F is not None:
    class TableDetectionModel(nn.Module):
        """
        Robust Table Detection Model for HTML.
        Combines a simple neural network for table structure classification
        with rule-based extraction using selectolax as a fallback.
        """

        def __init__(self, input_dim=128, hidden_dim=64, num_classes=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        @classmethod
        def load_from_checkpoint(cls, path=None):
            """
            Load the model from a PyTorch checkpoint.
            If no path is provided, use TABLE_MODEL_PATH from config.py.
            """
            if path is None:
                path = TABLE_MODEL_PATH
            checkpoint = torch.load(path, map_location="cpu")
            model = cls(**checkpoint.get("model_args", {}))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            return model

        def forward(self, x) -> Any:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def predict_tables(self, html: str) -> list[dict]:
            """
            Detect and extract tables from HTML using selectolax.
            Returns a list of dicts: [{"headers": [...], "data": [...], "meta": {...}}, ...]
            """
            tables = []
            tree = HTMLParser(html)

            for table in tree.css("table"):
                headers = []
                data = []
                meta = {}

                # Extract headers
                header_row = table.css_first("tr")
                if header_row:
                    headers = [cell.text(strip=True) for cell in header_row.css("th,td")]
                # Extract data rows
                for row in table.css("tr")[1:]:
                    cells = row.css("td,th")
                    row_data = {headers[i]: cells[i].text(strip=True) if i < len(cells) else "" for i in range(len(headers))}
                    if any(row_data.values()):
                        data.append(row_data)
                meta = {
                    "source": "selectolax_table",
                    "n_rows": len(data),
                    "n_cols": len(headers),
                    "table_html": table.html[:1000] if hasattr(table, "html") else ""
                }
                tables.append({"headers": headers, "data": data, "meta": meta})

            # Fallback: Regex-based detection for table-like structures
            if not tables:
                tables.extend(self._regex_table_detection(html))

            return tables

        def _regex_table_detection(self, html: str) -> list[dict]:
            """
            Fallback: Use regex to find repeated row/column patterns in flat HTML.
            Returns list of {headers, data, meta}.
            """
            tables = []
            lines = [line.strip() for line in html.splitlines() if line.strip()]
            col_counts = [len(re.split(r"\s{2,}|\t|\|", line)) for line in lines]
            if not col_counts:
                return []
            count_freq = Counter(col_counts)
            common_col = max(
                (count for count in count_freq if count > 1),
                key=lambda count: count_freq[count],
                default=None,
            )
            if not common_col or count_freq[common_col] < 2:
                return []
            rows = [
                re.split(r"\s{2,}|\t|\|", line)
                for line, col_count in zip(lines, col_counts)
                if col_count == common_col
            ]
            if len(rows) < 2:
                return []
            headers = rows[0]
            data = []
            for row in rows[1:]:
                row_data = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
                if any(row_data.values()):
                    data.append(row_data)
            meta = {
                "source": "regex_table",
                "n_rows": len(data),
                "n_cols": len(headers),
            }
            tables.append({"headers": headers, "data": data, "meta": meta})
            return tables
else:
    class TableDetectionModel:  # type: ignore[no-redef]
        @classmethod
        def load_from_checkpoint(cls, path=None):
            raise ImportError("Torch is not available: cannot load TableDetectionModel")

        def predict_tables(self, html: str) -> list[dict]:
            raise ImportError("Torch is not available: predict_tables unsupported")
            

# --- Example Usage ---

if __name__ == "__main__":
    # Example: Predict fields for a contest title
    test_title = "2022 General Election Los Angeles County California"
    try:
        model = ModelRegistry.get_torch_contest_model()
        result = model.predict(test_title)
        print("Prediction for:", test_title)
        for field, info in result.items():
            print(
                f"  {field}: {info['value']} (confidence: {info['confidence']:.2f}) -- {info['explanation']}"
            )
    except Exception as e:
        print(f"Error in model prediction: {e}")

# --- Recommendations for librarian.py integration ---
# To make vocabularies and context even more robust and context-aware:
# - Add functions in librarian.py to auto-generate and update vocab files from your context library and logs.
# - Use librarian.py to maintain mappings for all entities (states, counties, types, years, candidates, offices, etc.).
# - Add utilities to librarian.py for entity normalization, alias resolution, and context enrichment.
# - Consider exporting entity frequency statistics and co-occurrence data for smarter ML/NLP suggestions.