from __future__ import annotations

import re
from threading import RLock
from typing import Any

from .ml_telemetry import record_ml_event

_SPACY_LOCK = RLock()
_SPACY_NLP = None
_SPACY_SOURCE: str | None = None


def _load_spacy_model():
    global _SPACY_NLP, _SPACY_SOURCE
    with _SPACY_LOCK:
        if _SPACY_SOURCE is not None:
            return _SPACY_NLP

        model_name = "en_core_web_sm"
        try:
            import os

            model_name = os.environ.get("SPACY_MODEL", "en_core_web_sm")
        except Exception:
            pass

        try:
            import spacy  # type: ignore

            _SPACY_NLP = spacy.load(model_name, disable=["textcat"])
            _SPACY_SOURCE = "spacy"
            return _SPACY_NLP
        except Exception:
            _SPACY_NLP = None
            _SPACY_SOURCE = "rules"
            return None


def _rule_based_entities(text: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    if not text:
        return entities

    patterns = [
        ("YEAR", re.compile(r"\b(19|20)\d{2}\b")),
        ("PERCENT", re.compile(r"\b\d{1,3}(?:\.\d+)?%\b")),
        ("VOTES", re.compile(r"\b\d{1,3}(?:,\d{3})+\b")),
    ]

    for label, pattern in patterns:
        for match in pattern.finditer(text):
            entities.append(
                {
                    "text": match.group(0),
                    "label": label,
                    "start": match.start(),
                    "end": match.end(),
                    "source": "rules",
                }
            )

    return entities


def extract_training_entities(
    text: str,
    *,
    session_id: str | None = None,
    max_entities: int = 40,
) -> list[dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return []

    nlp = _load_spacy_model()
    entities: list[dict[str, Any]] = []

    if nlp is not None:
        try:
            doc = nlp(text)
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": int(ent.start_char),
                        "end": int(ent.end_char),
                        "source": "spacy",
                    }
                )
            record_ml_event(
                "nlp_entity_extractor",
                "spacy_inference",
                session_id=session_id,
                metadata={"text_len": len(text), "entities": len(entities)},
            )
        except Exception as exc:
            record_ml_event(
                "nlp_entity_extractor",
                "spacy_inference_failed",
                session_id=session_id,
                metadata={"error": str(exc), "text_len": len(text)},
            )
            entities = _rule_based_entities(text)
    else:
        entities = _rule_based_entities(text)
        record_ml_event(
            "nlp_entity_extractor",
            "rules_inference",
            session_id=session_id,
            metadata={"text_len": len(text), "entities": len(entities)},
        )

    # deterministic de-dup and cap
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for ent in entities:
        key = (ent.get("text"), ent.get("label"), ent.get("start"), ent.get("end"), ent.get("source"))
        dedup[key] = ent
    result = list(dedup.values())
    if len(result) > max_entities:
        result = result[:max_entities]

    record_ml_event(
        "nlp_entity_extractor",
        "entities_emitted",
        session_id=session_id,
        metadata={"count": len(result), "source": _SPACY_SOURCE or "rules"},
    )
    return result
