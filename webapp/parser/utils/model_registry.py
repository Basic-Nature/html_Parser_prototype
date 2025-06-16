"""
model_registry.py

Centralized registry and loader for all ML/NLP models used in the project.
Supports SentenceTransformer, spaCy, and custom models.
Ensures models are loaded once, cached, and reused across modules.
Integrates with config.py for model directory paths.
"""

import threading
import os
import logging

from ..config import MODEL_DIR

# Optional: Add more imports as needed for other models
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import spacy
except ImportError:
    spacy = None

logger = logging.getLogger("model_registry")
_lock = threading.Lock()

class ModelRegistry:
    """
    Centralized registry for ML/NLP models.
    Supports:
      - Fine-tuned and base SentenceTransformer models
      - Fine-tuned and base spaCy models
      - Arbitrary custom models (e.g., sklearn, torch, etc.)
    """
    _models = {}
    _nlp_models = {}
    _custom_models = {}
    _model_paths = {
        "sentence_transformer": os.path.join(MODEL_DIR, "fine_tuned_table_headers_tmp"),
        "spacy_ner": os.path.join(MODEL_DIR, "fine_tuned_spacy_ner"),
    }

    @classmethod
    def get_sentence_transformer(cls, model_name=None, use_finetuned=True):
        """
        Load and cache a SentenceTransformer model.
        If use_finetuned is True, tries to load the fine-tuned model from disk first.
        """
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers is not installed.")
        with _lock:
            key = f"sentence_transformer:{model_name or 'default'}:{use_finetuned}"
            if key in cls._models:
                return cls._models[key]
            # Try fine-tuned model first
            if use_finetuned:
                finetuned_path = cls._model_paths["sentence_transformer"]
                if os.path.exists(os.path.join(finetuned_path, "config.json")):
                    logger.info(f"Loading fine-tuned SentenceTransformer from {finetuned_path}")
                    model = SentenceTransformer(finetuned_path)
                    cls._models[key] = model
                    return model
            # Fallback to base model
            base_name = model_name or "all-MiniLM-L6-v2"
            logger.info(f"Loading base SentenceTransformer: {base_name}")
            model = SentenceTransformer(base_name)
            cls._models[key] = model
            return model

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
                if os.path.exists(os.path.join(finetuned_path, "meta.json")):
                    logger.info(f"Loading fine-tuned spaCy model from {finetuned_path}")
                    nlp = spacy.load(finetuned_path)
                    cls._nlp_models[key] = nlp
                    return nlp
            # Fallback to base model
            base_name = model_name or "en_core_web_sm"
            try:
                logger.info(f"Loading base spaCy model: {base_name}")
                nlp = spacy.load(base_name)
            except OSError:
                # Auto-download if missing
                import subprocess
                logger.info(f"Downloading spaCy model: {base_name}")
                subprocess.run(["python", "-m", "spacy", "download", base_name], check=True)
                nlp = spacy.load(base_name)
            cls._nlp_models[key] = nlp
            return nlp

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
                cls._custom_models[key] = loader_func(*args, **kwargs)
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
            elif model_type == "spacy":
                key_prefix = f"spacy:{model_name or 'default'}"
                cls._nlp_models = {k: v for k, v in cls._nlp_models.items() if not k.startswith(key_prefix)}
            else:
                if model_type in cls._custom_models:
                    del cls._custom_models[model_type]
            logger.info(f"Reloaded model(s) of type: {model_type}")

    @classmethod
    def clear_cache(cls):
        """Clear all loaded models (for testing or reloading)."""
        with _lock:
            cls._models.clear()
            cls._nlp_models.clear()
            cls._custom_models.clear()
            logger.info("Model registry cache cleared.")

    @classmethod
    def set_model_path(cls, model_type, path):
        """Set or override the path for a given model type."""
        cls._model_paths[model_type] = path

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