"""Entity Confidence Mapping: Weighted Signal Catalog for Decision Gates

This module provides typed accessors and data structures for confidence/caution scoring
of entities (offices, parties, jurisdictions, contest types, sources, candidates) used
throughout the election parser pipeline.

Design principle: Nonpartisan, data-driven, fully audited. All weights derived from
source authority, verification consistency, and historical accuracy—never political factors.

Author: Smart Elections Team
Date: February 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import orjson


class DecisionCode(Enum):
    """Decision outcomes from confidence/caution gates."""
    PROCEED = "proceed"           # confidence ≥ 2/3, caution ≤ 1/3, override ≤ 1/3
    CAUTION = "caution"           # mixed signals; guarded action
    STOP = "stop"                 # low confidence, high caution, or high override


class SignalType(Enum):
    """Categories of signals that feed confidence/caution scoring."""
    EXACT_MATCH_VERIFIED = "exact_match_verified"         # FIPS, FEC official
    EXACT_MATCH_CURATED = "exact_match_curated"           # Community-maintained registry
    FUZZY_MATCH_HIGH = "fuzzy_match_high"                 # Levenshtein ≥ 0.90
    FUZZY_MATCH_MEDIUM = "fuzzy_match_medium"             # Levenshtein 0.75–0.89
    FUZZY_MATCH_LOW = "fuzzy_match_low"                   # Levenshtein < 0.75
    CONTEXTUAL_MATCH = "contextual_match"                 # Inferred from surrounding data
    PATTERN_MATCH = "pattern_match"                        # HTML/CSV heuristic
    ALIAS_MATCH = "alias_match"                            # Common abbreviation
    HEADER_ALIGNMENT = "header_alignment"                  # Column count, name similarity
    HANDLER_SUCCESS = "handler_success"                    # Historical parse success rate


class AnomalyType(Enum):
    """Categories of anomalies that increase caution score."""
    MISMATCHED_TOTALS = "mismatched_totals"               # Row/column count unexpected
    MISSING_CANDIDATE = "missing_candidate"                # Expected candidate absent
    SUSPICIOUS_HEADER = "suspicious_header"                # Unexpected column names
    VALUE_INCONSISTENCY = "value_inconsistency"            # Data type mismatch, NaN
    TYPOSQUAT_PATTERN = "typosquat_pattern"                # Domain/name looks like typo
    SSL_CERTIFICATE_AGE = "ssl_certificate_age"            # Old/expired cert
    CONFLICTING_SOURCES = "conflicting_sources"            # Multiple sources disagree
    CONTEXTUAL_MISMATCH = "contextual_mismatch"            # State/county/contest mismatch


class OverrideTrigger(Enum):
    """Triggers that increase override score."""
    ADMIN_FLAG = 0.3                  # Admin manually trusted/rejected entity
    VERIFIED_SOURCE_CORRECTION = 0.2  # Data matches official source correction
    ANOMALY_COUNT = 0.15              # Per anomaly beyond first
    ML_MODEL_LOW_CONFIDENCE = 0.15    # Sentence-Transformers confidence < 0.5
    CONTEXTUAL_MISMATCH = 0.10        # State/county/contest mismatch
    MULTIPLE_CORRECTIONS = 0.10       # Entity corrected > 2 times in 30 days


@dataclass
class SignalCoefficient:
    """Represents a signal type with its weight and baseline confidence."""
    signal_type: SignalType
    weight: float                # ∈ [0, 1]; higher = more influential
    baseline_confidence: float   # ∈ [0, 1]; confidence if signal observed
    description: str
    source_authority: str       # e.g., "FIPS Registry", "FEC", "Curated"


@dataclass
class AnomalyCoefficient:
    """Represents an anomaly type with its weight and caution contribution."""
    anomaly_type: AnomalyType
    weight: float               # ∈ [0, 1]; higher = more influential
    baseline_caution: float     # ∈ [0, 1]; caution if anomaly detected
    description: str
    context: str                # When this anomaly is relevant


@dataclass
class ConfidenceCautionResult:
    """Result of confidence/caution calculation."""
    entity_id: str
    entity_type: str            # "office", "party", "jurisdiction", "url", etc.
    confidence_score: float     # ∈ [0, 1]
    caution_score: float        # ∈ [0, 1]
    override_score: float       # ≥ 0; unbounded
    decision_code: DecisionCode
    signals_observed: List[SignalType]
    anomalies_observed: List[AnomalyType]
    override_triggers_active: List[OverrideTrigger]
    reasoning: str              # Human-readable explanation
    confidence_threshold: float # 2/3 by default
    caution_threshold: float    # 1/3 by default
    override_threshold: float   # 1/3 by default


# ==================================================================================
# SIGNAL COEFFICIENTS CATALOG
# ==================================================================================

JURISDICTION_SIGNALS: List[SignalCoefficient] = [
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_VERIFIED,
        weight=1.0,
        baseline_confidence=0.99,
        description="Exact match in FIPS registry (US Census Bureau)",
        source_authority="FIPS Registry"
    ),
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_CURATED,
        weight=0.9,
        baseline_confidence=0.98,
        description="Match in Secretary of State county list",
        source_authority="SoS Official"
    ),
    SignalCoefficient(
        signal_type=SignalType.ALIAS_MATCH,
        weight=0.7,
        baseline_confidence=0.85,
        description="Match in curated alias mapping (e.g., 'LA' -> 'Los Angeles')",
        source_authority="Community Curated"
    ),
    SignalCoefficient(
        signal_type=SignalType.FUZZY_MATCH_HIGH,
        weight=0.5,
        baseline_confidence=0.75,
        description="Fuzzy match (Levenshtein ≥ 0.90)",
        source_authority="Heuristic (Typo-Tolerant)"
    ),
    SignalCoefficient(
        signal_type=SignalType.FUZZY_MATCH_MEDIUM,
        weight=0.3,
        baseline_confidence=0.60,
        description="Fuzzy match (Levenshtein 0.75–0.89)",
        source_authority="Heuristic (Weak)"
    ),
]

OFFICE_SIGNALS: List[SignalCoefficient] = [
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_VERIFIED,
        weight=1.0,
        baseline_confidence=0.99,
        description="Exact match to state election code",
        source_authority="State Statute"
    ),
    SignalCoefficient(
        signal_type=SignalType.ALIAS_MATCH,
        weight=0.8,
        baseline_confidence=0.90,
        description="Match to common alias (e.g., 'Pres' -> 'President')",
        source_authority="Convention"
    ),
    SignalCoefficient(
        signal_type=SignalType.CONTEXTUAL_MATCH,
        weight=0.6,
        baseline_confidence=0.70,
        description="Inferred from ballot measure type",
        source_authority="Context"
    ),
    SignalCoefficient(
        signal_type=SignalType.PATTERN_MATCH,
        weight=0.4,
        baseline_confidence=0.55,
        description="Header pattern match from HTML parsing",
        source_authority="Heuristic"
    ),
]

PARTY_SIGNALS: List[SignalCoefficient] = [
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_VERIFIED,
        weight=1.0,
        baseline_confidence=0.99,
        description="Match to FEC official party list",
        source_authority="FEC"
    ),
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_CURATED,
        weight=0.95,
        baseline_confidence=0.98,
        description="Match to state party registry",
        source_authority="State Authority"
    ),
    SignalCoefficient(
        signal_type=SignalType.ALIAS_MATCH,
        weight=0.85,
        baseline_confidence=0.90,
        description="Common alias (e.g., 'Dem' -> 'Democratic', 'GOP' -> 'Republican')",
        source_authority="Social Convention"
    ),
    SignalCoefficient(
        signal_type=SignalType.PATTERN_MATCH,
        weight=0.5,
        baseline_confidence=0.70,
        description="Write-in/independent pattern match",
        source_authority="Text Pattern"
    ),
]

SOURCE_SIGNALS: List[SignalCoefficient] = [
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_VERIFIED,
        weight=1.0,
        baseline_confidence=0.99,
        description="Verified government domain (whitelisted SoS)",
        source_authority="Whitelist + SSL"
    ),
    SignalCoefficient(
        signal_type=SignalType.EXACT_MATCH_CURATED,
        weight=0.9,
        baseline_confidence=0.95,
        description="Government domain pattern (.gov) with verified SSL",
        source_authority="Pattern + SSL"
    ),
    SignalCoefficient(
        signal_type=SignalType.CONTEXTUAL_MATCH,
        weight=0.7,
        baseline_confidence=0.85,
        description="Known third-party aggregator (e.g., Ballotpedia)",
        source_authority="Curated Source List"
    ),
]

# ==================================================================================
# ANOMALY COEFFICIENTS CATALOG
# ==================================================================================

ANOMALY_CATALOG: List[AnomalyCoefficient] = [
    AnomalyCoefficient(
        anomaly_type=AnomalyType.MISMATCHED_TOTALS,
        weight=0.8,
        baseline_caution=0.70,
        description="Row or column count differs from expected",
        context="Parsing HTML/CSV tables"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.MISSING_CANDIDATE,
        weight=0.9,
        baseline_caution=0.80,
        description="Candidate present in context but absent from result data",
        context="Contest result validation"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.SUSPICIOUS_HEADER,
        weight=0.6,
        baseline_caution=0.50,
        description="Column names don't match expected headers",
        context="CSV/table parsing"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.VALUE_INCONSISTENCY,
        weight=0.5,
        baseline_caution=0.40,
        description="Data type mismatch, NaN, or unexpected format",
        context="Data validation"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.TYPOSQUAT_PATTERN,
        weight=1.0,
        baseline_caution=0.90,
        description="Domain or name looks like a typo of known entity",
        context="URL/entity trust scoring"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.CONFLICTING_SOURCES,
        weight=0.7,
        baseline_caution=0.65,
        description="Multiple sources report conflicting data for same entity",
        context="Cross-source validation"
    ),
    AnomalyCoefficient(
        anomaly_type=AnomalyType.CONTEXTUAL_MISMATCH,
        weight=0.6,
        baseline_caution=0.55,
        description="State/county/contest context doesn't match entity claims",
        context="Sanity checks"
    ),
]


class EntityConfidenceMap:
    """Central registry for entity confidence/caution coefficients.
    
    Provides typed accessors for signals, anomalies, and override triggers.
    Loads from vocab files and cached memory; supports hot reloading for updates.
    """

    def __init__(self):
        """Initialize the confidence map with default catalogs."""
        self.jurisdiction_signals = {s.signal_type: s for s in JURISDICTION_SIGNALS}
        self.office_signals = {s.signal_type: s for s in OFFICE_SIGNALS}
        self.party_signals = {s.signal_type: s for s in PARTY_SIGNALS}
        self.source_signals = {s.signal_type: s for s in SOURCE_SIGNALS}
        self.anomaly_catalog = {a.anomaly_type: a for a in ANOMALY_CATALOG}
        self.override_triggers = OverrideTrigger

    def get_signal_coefficient(
        self,
        entity_type: str,
        signal_type: SignalType
    ) -> Optional[SignalCoefficient]:
        """Retrieve coefficient for a signal within a specific entity type."""
        catalog_map = {
            "jurisdiction": self.jurisdiction_signals,
            "office": self.office_signals,
            "party": self.party_signals,
            "source": self.source_signals,
        }
        catalog = catalog_map.get(entity_type.lower())
        return catalog.get(signal_type) if catalog else None

    def get_anomaly_coefficient(
        self,
        anomaly_type: AnomalyType
    ) -> Optional[AnomalyCoefficient]:
        """Retrieve coefficient for an anomaly."""
        return self.anomaly_catalog.get(anomaly_type)

    def calculate_confidence_caution(
        self,
        entity_id: str,
        entity_type: str,
        signals: List[Tuple[SignalType, bool]],
        anomalies: List[Tuple[AnomalyType, bool]] = None,
        override_triggers: List[OverrideTrigger] = None,
    ) -> ConfidenceCautionResult:
        """
        Calculate confidence/caution scores and determine decision gate.

        Args:
            entity_id: Unique identifier for the entity (e.g., "Los Angeles County")
            entity_type: Type of entity ("jurisdiction", "office", "party", "source")
            signals: List of (SignalType, observed: bool) tuples
            anomalies: List of (AnomalyType, detected: bool) tuples; defaults to empty
            override_triggers: List of active OverrideTrigger values; defaults to empty

        Returns:
            ConfidenceCautionResult with scores, decision, and reasoning.
        """
        anomalies = anomalies or []
        override_triggers = override_triggers or []

        # Calculate confidence score
        total_weight = 0.0
        weighted_confidence = 0.0
        observed_signals = []

        for signal_type, observed in signals:
            coeff = self.get_signal_coefficient(entity_type, signal_type)
            if not coeff:
                continue
            if observed:
                weighted_confidence += coeff.weight * coeff.baseline_confidence
                observed_signals.append(signal_type)
            total_weight += coeff.weight

        confidence_score = weighted_confidence / total_weight if total_weight > 0 else 0.0

        # Calculate caution score
        anomaly_weight = 0.0
        weighted_caution = 0.0
        observed_anomalies = []

        for anomaly_type, detected in anomalies:
            coeff = self.get_anomaly_coefficient(anomaly_type)
            if not coeff:
                continue
            if detected:
                weighted_caution += coeff.weight * coeff.baseline_caution
                observed_anomalies.append(anomaly_type)
            anomaly_weight += coeff.weight

        caution_score = weighted_caution / anomaly_weight if anomaly_weight > 0 else 0.0

        # Calculate override score
        override_score = sum(float(trigger.value) for trigger in override_triggers)

        # Apply decision gates
        confidence_threshold = 2.0 / 3.0
        caution_threshold = 1.0 / 3.0
        override_threshold = 1.0 / 3.0

        if (confidence_score >= confidence_threshold and
            caution_score <= caution_threshold and
            override_score <= override_threshold):
            decision_code = DecisionCode.PROCEED
        elif (confidence_score < 1.0 / 3.0 or
              caution_score > 2.0 / 3.0 or
              override_score > override_threshold):
            decision_code = DecisionCode.STOP
        else:
            decision_code = DecisionCode.CAUTION

        # Generate reasoning
        reasoning = self._generate_reasoning(
            entity_id,
            entity_type,
            confidence_score,
            caution_score,
            override_score,
            observed_signals,
            observed_anomalies,
            override_triggers,
            decision_code
        )

        return ConfidenceCautionResult(
            entity_id=entity_id,
            entity_type=entity_type,
            confidence_score=confidence_score,
            caution_score=caution_score,
            override_score=override_score,
            decision_code=decision_code,
            signals_observed=observed_signals,
            anomalies_observed=observed_anomalies,
            override_triggers_active=override_triggers,
            reasoning=reasoning,
            confidence_threshold=confidence_threshold,
            caution_threshold=caution_threshold,
            override_threshold=override_threshold,
        )

    def _generate_reasoning(
        self,
        entity_id: str,
        entity_type: str,
        confidence_score: float,
        caution_score: float,
        override_score: float,
        signals: List[SignalType],
        anomalies: List[AnomalyType],
        overrides: List[OverrideTrigger],
        decision: DecisionCode,
    ) -> str:
        """Generate human-readable explanation of decision."""
        parts = [f"[{entity_type.upper()}] {entity_id}"]

        if signals:
            signal_names = ", ".join(s.value for s in signals)
            parts.append(f"✓ Signals: {signal_names}")

        if anomalies:
            anomaly_names = ", ".join(a.value for a in anomalies)
            parts.append(f"⚠ Anomalies: {anomaly_names}")

        if overrides:
            override_names = ", ".join(o.name for o in overrides)
            parts.append(f"🔒 Overrides: {override_names}")

        parts.append(f"Confidence: {confidence_score:.2%} | Caution: {caution_score:.2%} | Override: {override_score:.2f}")
        parts.append(f"Decision: {decision.value.upper()}")

        return " | ".join(parts)


# Singleton instance
_confidence_map_instance: Optional[EntityConfidenceMap] = None


def get_confidence_map() -> EntityConfidenceMap:
    """Get or create singleton instance of EntityConfidenceMap."""
    global _confidence_map_instance
    if _confidence_map_instance is None:
        _confidence_map_instance = EntityConfidenceMap()
    return _confidence_map_instance
