"""
model_registry.py

Centralized registry and loader for all ML/NLP models used in the project.
Supports SentenceTransformer, spaCy, and custom models.
Ensures models are loaded once, cached, and reused across modules.
Integrates with config.py for model directory paths.
Optimized for robust, singleton-style loading, device selection, path validation, and logging.
"""

import threading
import os
import sys
import re
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.shared_logger import SharedLogger
from ..config import MODEL_DIR, PROJECT_ROOT, VOCAB_DIR

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    spacy = None
    Language = None
logger = SharedLogger()
_lock = threading.Lock()

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

def advanced_tokenizer(text: str, max_len: int = 20) -> torch.Tensor:
    """
    Tokenizes text using the project vocabulary.
    Handles lowercasing, punctuation, and OOV words.
    Pads/truncates to max_len.
    """
    tokens = re.findall(r"\w+", text.lower())
    idxs = [WORD2IDX.get(tok, 0) for tok in tokens]
    idxs = idxs[:max_len] + [0] * (max_len - len(idxs))
    return torch.tensor(idxs).unsqueeze(0)  # (1, max_len)

# --- ContestFieldClassifier (Multi-head, Multi-field) ---

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
        if spacy is None:
            raise ImportError("spaCy is not installed.")
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
                import subprocess
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
        if cls._torch_candidate_model is not None:
            return cls._torch_candidate_model

        # --- Dynamic vocab loading from librarian.py ---
        try:
            from ..bots.librarian import load_context_library
            context = load_context_library()
            candidate_vocab = context.get("candidate_keywords", [])
            CANDIDATE2IDX = {c: i+1 for i, c in enumerate(sorted(set(candidate_vocab)))}
            IDX2CANDIDATE = {v: k for k, v in CANDIDATE2IDX.items()}
        except Exception as e:
            logger.error(f"Failed to load candidate vocab from librarian.py: {e}")
            CANDIDATE2IDX = {}
            IDX2CANDIDATE = {}

        # --- Model loading ---
        model_path = cls._model_paths.get("torch_candidate", os.path.join(MODEL_DIR, "candidate_classifier.pt"))
        if not os.path.exists(model_path):
            logger.error(f"Torch candidate model checkpoint not found: {model_path}")
            return None

        try:
            logger.info(f"Loading CandidateClassifier from {model_path}")
            model = CandidateClassifier.load_from_checkpoint(
                model_path,
                vocab_size=max(CANDIDATE2IDX.values(), default=1) + 1,
                embed_dim=128,
                num_candidates=max(CANDIDATE2IDX.values(), default=1) + 1
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
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
        with _lock:
            base_name = model_name or "all-MiniLM-L6-v2"
            if not isinstance(base_name, str) or base_name.strip() == "" or base_name.startswith("SentenceTransformer("):
                logger.error(f"Invalid model_name for SentenceTransformer: {base_name!r}. Using default 'all-MiniLM-L6-v2'.")
                base_name = "all-MiniLM-L6-v2"
            key = f"sentence_transformer:{base_name}:{use_finetuned}"
            if key in cls._models:
                return cls._models[key]
            # Try fine-tuned model first
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
            # Fallback to base model
            logger.info(f"Loading base SentenceTransformer: {base_name}")
            try:
                model = SentenceTransformer(base_name, device=device or cls._get_device())
                cls._models[key] = model
                cls._loaded_info[key] = base_name
                return model
            except Exception as e:
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

# --- Example Usage ---

if __name__ == "__main__":
    # Example: Predict fields for a contest title
    try:
        model = ModelRegistry.get_torch_contest_model()
        test_title = "2022 General Election Los Angeles County California"
        result = model.predict(test_title)
        print("Prediction for:", test_title)
        for field, info in result.items():
            print(f"  {field}: {info['value']} (confidence: {info['confidence']:.2f}) -- {info['explanation']}")
    except Exception as e:
        print(f"Error in model prediction: {e}")

# --- Recommendations for librarian.py integration ---
# To make vocabularies and context even more robust and context-aware:
# - Add functions in librarian.py to auto-generate and update vocab files from your context library and logs.
# - Use librarian.py to maintain mappings for all entities (states, counties, types, years, candidates, offices, etc.).
# - Add utilities to librarian.py for entity normalization, alias resolution, and context enrichment.
# - Consider exporting entity frequency statistics and co-occurrence data for smarter ML/NLP suggestions.