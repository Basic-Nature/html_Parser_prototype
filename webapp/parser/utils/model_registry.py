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
import logging
import sys

from ..config import MODEL_DIR, PROJECT_ROOT

try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    SentenceTransformer = None
    torch = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger("model_registry")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)
_lock = threading.Lock()

class ModelRegistry:
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
    }
    _loaded_info = {}

    @classmethod
    def _get_device(cls):
        if torch is not None and torch.cuda.is_available():
            logger.info("Using CUDA for model loading.")
            return "cuda"
        logger.info("Using CPU for model loading.")
        return "cpu"

    @classmethod
    def get_sentence_transformer(cls, model_name=None, use_finetuned=True, device=None):
        """
        Load and cache a SentenceTransformer model.
        If use_finetuned is True, tries to load the fine-tuned model from disk first.
        Ensures only one instance per model is loaded (efficient caching).
        Optimized to prevent invalid model names/paths and repeated failed loads.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
        with _lock:
            base_name = model_name or "all-MiniLM-L6-v2"
            # Validate model_name: must be a string and not a model object or repr
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
                        # Do not retry with the same key if it failed
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
                # Do not retry with the same key if it failed
                cls._models[key] = None
                return None

    @classmethod
    def get_spacy_model(cls, model_name=None, use_finetuned=True):
        """
        Load and cache a spaCy model.
        If use_finetuned is True, tries to load the fine-tuned model from disk first.
        """
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
    def get_custom_model(cls, key, loader_func, *args, **kwargs):
        """
        Load and cache a custom model using a loader function.
        Example:
            def load_my_model(path): ...
            model = ModelRegistry.get_custom_model("my_model", load_my_model, path)
        """
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
    def reload_model(cls, model_type, model_name=None):
        """
        Force reload a model (e.g., after retraining).
        model_type: 'sentence_transformer', 'spacy', or custom key.
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
    def clear_cache(cls):
        """Clear all loaded models (for testing or reloading)."""
        with _lock:
            cls._models.clear()
            cls._nlp_models.clear()
            cls._custom_models.clear()
            cls._loaded_info.clear()
            logger.info("Model registry cache cleared.")

    @classmethod
    def set_model_path(cls, model_type, path):
        """Set or override the path for a given model type."""
        cls._model_paths[model_type] = path

    @classmethod
    def get_model_name(cls, model):
        """
        Return the model name string for a SentenceTransformer instance.
        """
        if hasattr(model, 'model_name_or_path'):
            return getattr(model, 'model_name_or_path')
        if hasattr(model, 'modules') and hasattr(model.modules[0], 'model_name_or_path'):
            return getattr(model.modules[0], 'model_name_or_path')
        return str(model)

    @classmethod
    def get_loaded_models_info(cls):
        """
        Return a dict of loaded model keys and their source paths/names.
        """
        return dict(cls._loaded_info)

# Example usage:
if __name__ == "__main__":
    # SentenceTransformer example
    try:
        st_model = ModelRegistry.get_sentence_transformer()
        print("Loaded SentenceTransformer:", st_model)
    except Exception as e:
        print("SentenceTransformer error:", e)

    # spaCy example
    try:
        nlp = ModelRegistry.get_spacy_model()
        print("Loaded spaCy model:", nlp)
    except Exception as e:
        print("spaCy error:", e)

    # Custom model example
    def dummy_loader():
        return {"model": "dummy"}
    dummy = ModelRegistry.get_custom_model("dummy", dummy_loader)
    print("Loaded custom model:", dummy)

    # Print loaded models info
    print("Loaded models info:", ModelRegistry.get_loaded_models_info())
