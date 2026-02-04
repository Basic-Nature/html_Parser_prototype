"""Safe Decision Helpers: Confidence/Caution Gates for Election Entities

This module provides typed safe_decide_* functions for guarded decision-making
throughout the parsing pipeline. All decisions logged to JSONL for audit.

Functions return DecisionTuple with confidence_score, caution_score, decision_code,
and reasoning. Three gates: PROCEED (full trust), CAUTION (guarded), STOP (reject).

Design principle: Nonpartisan, data-driven, fully audited.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from webapp.parser.Context_Integration.library.entity_confidence_map import (
    AnomalyType,
    DecisionCode,
    EntityConfidenceMap,
    OverrideTrigger,
    SignalType,
    get_confidence_map,
)
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.shared_logic import DecisionTuple


def _emit_decision_log(
    decision_tuple: DecisionTuple,
    session_id: Optional[str] = None,
) -> None:
    """Log decision to structured JSONL for audit trail.
    
    Format:
    {
        "event_type": "decision",
        "decision_code": "proceed|caution|stop",
        "entity_id": "...",
        "entity_type": "office|party|jurisdiction|source",
        "confidence_score": 0.0-1.0,
        "caution_score": 0.0-1.0,
        "override_score": 0.0+,
        "signals_used": [...],
        "anomalies_detected": [...],
        "timestamp": ISO8601,
        "session_id": optional,
    }
    """
    try:
        payload = {
            "level": "INFO",
            "type": "decision",
            "message": decision_tuple.get("reasoning", "decision"),
            "event_type": "decision",
            "decision_code": decision_tuple.get("decision_code"),
            "confidence_score": decision_tuple.get("confidence_score"),
            "caution_score": decision_tuple.get("caution_score"),
            "override_score": decision_tuple.get("override_score"),
            "signals_observed": decision_tuple.get("signals_observed", []),
            "anomalies_observed": decision_tuple.get("anomalies_observed", []),
            "timestamp": decision_tuple.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "session_id": session_id,
        }
        logger.info(payload)
    except Exception as e:
        try:
            logger.warning({
                "level": "WARNING",
                "type": "decision",
                "message": f"Failed to log decision: {e}",
                "session_id": session_id,
            })
        except Exception:
            pass


def safe_decide_jurisdiction(
    entity_id: str,
    state: str,
    signals: List[Tuple[SignalType, bool]],
    anomalies: List[Tuple[AnomalyType, bool]] = None,
    overrides: List[OverrideTrigger] = None,
    session_id: Optional[str] = None,
) -> DecisionTuple:
    """
    Decide on a jurisdiction (county, city, state).
    
    Args:
        entity_id: County/jurisdiction name
        state: State abbreviation
        signals: [(SignalType, observed), ...] 
        anomalies: [(AnomalyType, detected), ...]
        overrides: [OverrideTrigger, ...]
        session_id: For audit linking
        
    Returns:
        DecisionTuple with decision_code in {"proceed", "caution", "stop"}
    """
    confidence_map = get_confidence_map()
    result = confidence_map.calculate_confidence_caution(
        entity_id=entity_id,
        entity_type="jurisdiction",
        signals=signals,
        anomalies=anomalies or [],
        override_triggers=overrides or [],
    )
    
    decision_tuple: DecisionTuple = {
        "value": entity_id,
        "decision_code": result.decision_code.value,
        "confidence_score": result.confidence_score,
        "caution_score": result.caution_score,
        "override_score": result.override_score,
        "signals_observed": [s.value for s in result.signals_observed],
        "anomalies_observed": [a.value for a in result.anomalies_observed],
        "reasoning": result.reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
    }
    
    _emit_decision_log(decision_tuple, session_id)
    return decision_tuple


def safe_decide_office(
    entity_id: str,
    state: str,
    signals: List[Tuple[SignalType, bool]],
    anomalies: List[Tuple[AnomalyType, bool]] = None,
    overrides: List[OverrideTrigger] = None,
    session_id: Optional[str] = None,
) -> DecisionTuple:
    """Decide on an office (President, Senator, etc.)."""
    confidence_map = get_confidence_map()
    result = confidence_map.calculate_confidence_caution(
        entity_id=entity_id,
        entity_type="office",
        signals=signals,
        anomalies=anomalies or [],
        override_triggers=overrides or [],
    )
    
    decision_tuple: DecisionTuple = {
        "value": entity_id,
        "decision_code": result.decision_code.value,
        "confidence_score": result.confidence_score,
        "caution_score": result.caution_score,
        "override_score": result.override_score,
        "signals_observed": [s.value for s in result.signals_observed],
        "anomalies_observed": [a.value for a in result.anomalies_observed],
        "reasoning": result.reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
    }
    
    _emit_decision_log(decision_tuple, session_id)
    return decision_tuple


def safe_decide_party(
    entity_id: str,
    signals: List[Tuple[SignalType, bool]],
    anomalies: List[Tuple[AnomalyType, bool]] = None,
    overrides: List[OverrideTrigger] = None,
    session_id: Optional[str] = None,
) -> DecisionTuple:
    """Decide on a political party (Democratic, Republican, etc.)."""
    confidence_map = get_confidence_map()
    result = confidence_map.calculate_confidence_caution(
        entity_id=entity_id,
        entity_type="party",
        signals=signals,
        anomalies=anomalies or [],
        override_triggers=overrides or [],
    )
    
    decision_tuple: DecisionTuple = {
        "value": entity_id,
        "decision_code": result.decision_code.value,
        "confidence_score": result.confidence_score,
        "caution_score": result.caution_score,
        "override_score": result.override_score,
        "signals_observed": [s.value for s in result.signals_observed],
        "anomalies_observed": [a.value for a in result.anomalies_observed],
        "reasoning": result.reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
    }
    
    _emit_decision_log(decision_tuple, session_id)
    return decision_tuple


def safe_decide_source(
    url: str,
    signals: List[Tuple[SignalType, bool]],
    anomalies: List[Tuple[AnomalyType, bool]] = None,
    overrides: List[OverrideTrigger] = None,
    session_id: Optional[str] = None,
) -> DecisionTuple:
    """Decide on a data source (URL, file, etc.)."""
    confidence_map = get_confidence_map()
    result = confidence_map.calculate_confidence_caution(
        entity_id=url,
        entity_type="source",
        signals=signals,
        anomalies=anomalies or [],
        override_triggers=overrides or [],
    )
    
    decision_tuple: DecisionTuple = {
        "value": url,
        "decision_code": result.decision_code.value,
        "confidence_score": result.confidence_score,
        "caution_score": result.caution_score,
        "override_score": result.override_score,
        "signals_observed": [s.value for s in result.signals_observed],
        "anomalies_observed": [a.value for a in result.anomalies_observed],
        "reasoning": result.reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
    }
    
    _emit_decision_log(decision_tuple, session_id)
    return decision_tuple


def should_proceed(decision_tuple: DecisionTuple) -> bool:
    """Check if decision is PROCEED (proceed=True, caution/stop=False)."""
    return decision_tuple.get("decision_code") == "proceed"


def should_caution(decision_tuple: DecisionTuple) -> bool:
    """Check if decision is CAUTION."""
    return decision_tuple.get("decision_code") == "caution"


def should_stop(decision_tuple: DecisionTuple) -> bool:
    """Check if decision is STOP."""
    return decision_tuple.get("decision_code") == "stop"
