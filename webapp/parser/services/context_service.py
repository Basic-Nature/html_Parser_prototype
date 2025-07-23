import os
import json
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from ..bots.librarian import (
    load_context_library,
    KNOWN_STATE_TO_COUNTY_MAP,
    KNOWN_COUNTY_TO_PRECINCTS_MAP,
    STATE_ABBR,
    CONTEST_KEYWORDS,
    PARTY_KEYWORDS,
    CANDIDATE_KEYWORDS,
    ELECTION_TYPES,
    # Add more normalization/alias utilities as needed
)
from ..utils.shared_logic import (
    PredictionResult, safe_append,
    normalize_state_name, normalize_county_name, resolve_county_alias
)
from ..utils.shared_logger import SharedLogger
from ..services.election_data_services import ElectionDataService
from ..utils.spacy_utils import extract_entities, extract_dates, extract_locations
from ..utils.user_prompt import UserPrompt

logger = SharedLogger()
AUDIT_LOG = "context_audit_log.jsonl"

class ContextBasedPredictor:
    def __init__(self, vocab_dir: str):
        self.context_service = ContextService(vocab_dir)
        self.db_service = ElectionDataService()
        self.state_to_county_map = KNOWN_STATE_TO_COUNTY_MAP
        # Optionally load ML/NLP models here (e.g., SentenceTransformer)
        try:
            from ..utils.model_registry import ModelRegistry
            self.embedding_model = ModelRegistry.get_sentence_transformer("all-MiniLM-L6-v2")
        except Exception:
            self.embedding_model = None

    def predict(self, text: str) -> PredictionResult:
        result: PredictionResult = {}
        text_norm = text.lower().strip() if text else ""

        # --- NLP Entity Extraction ---
        entities = extract_entities(text)
        dates = extract_dates(text)
        locations = extract_locations(text)

        # --- State ---
        state = self.context_service.normalize_state(text)
        if not state:
            state = next((normalize_state_name(ent) for ent, label in locations if label in {"GPE", "LOC"}), None)
        if state:
            result["state"] = state
            result["state_abbr"] = next((abbr for abbr, s in STATE_ABBR.items() if s == state), None)

        # --- County (with validation using KNOWN_STATE_TO_COUNTY_MAP) ---
        county = resolve_county_alias(text, state)
        if not county:
            county = next((normalize_county_name(ent) for ent, label in locations if label in {"GPE", "LOC"}), None)
        if county and state:
            valid_counties = self.state_to_county_map.get(state, [])
            if valid_counties and county not in valid_counties:
                # Fuzzy match or fallback to best match
                import difflib
                matches = difflib.get_close_matches(county, valid_counties, n=1, cutoff=0.7)
                if matches:
                    county = matches[0]
        if county:
            result["county"] = county

        # --- Precinct ---
        precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP.get(county, [])
        if precincts:
            for p in precincts:
                if p in text_norm:
                    result["precinct"] = p
                    break

        # --- Year ---
        year_match = re.search(r"\b(20\d{2})\b", text_norm)
        if year_match:
            result["year"] = int(year_match.group(1))
        else:
            year_from_nlp = next((int(ent) for ent, label in dates if re.match(r"\b(20\d{2})\b", ent)), None)
            if year_from_nlp:
                result["year"] = year_from_nlp

        # --- Election Type ---
        result["election_types"] = [etype for etype in ELECTION_TYPES if etype in text_norm]
        if result["election_types"]:
            result["type_"] = result["election_types"][0]
        else:
            etype_nlp = next((etype for etype in ELECTION_TYPES for ent, label in entities if etype in ent.lower()), None)
            if etype_nlp:
                result["type_"] = etype_nlp
                result["election_types"] = [etype_nlp]

        # --- Office ---
        office = next((kw for kw in CONTEST_KEYWORDS if kw in text_norm), None)
        if not office:
            office = next((ent for ent, label in entities if label in {"ORG", "NORP"}), None)
        if office:
            result["office"] = office

        # --- Party ---
        party = next((kw for kw in PARTY_KEYWORDS if kw in text_norm), None)
        if not party:
            party = next((ent for ent, label in entities if label == "NORP"), None)
        if party:
            result["party"] = party

        # --- Candidate ---
        candidate = next((kw for kw in CANDIDATE_KEYWORDS if kw in text_norm), None)
        if not candidate:
            candidate = next((ent for ent, label in entities if label in {"PERSON", "CANDIDATE"}), None)
        if candidate:
            result["candidate"] = candidate

        # --- Ballot Type ---
        ballot_type_match = re.search(r"(absentee|mail|early voting|provisional|affidavit|election day)", text_norm)
        if ballot_type_match:
            result["ballot_type"] = ballot_type_match.group(1)

        # --- Vote Method ---
        vote_method_match = re.search(r"(in-person|mail|drop box|online|provisional)", text_norm)
        if vote_method_match:
            result["vote_method"] = vote_method_match.group(1)

        # --- Timestamp ---
        timestamp_match = re.search(r"\b(20\d{2}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2})\b", text)
        if timestamp_match:
            result["timestamp"] = timestamp_match.group(1)
        else:
            timestamp_nlp = next((ent for ent, label in dates if re.match(r"\b(20\d{2}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2})\b", ent)), None)
            if timestamp_nlp:
                result["timestamp"] = timestamp_nlp

        # --- Source URL (if present in text) ---
        url_match = re.search(r"https?://[^\s]+", text)
        if url_match:
            result["source_url"] = url_match.group(0)

        # --- DB Lookup for enrichment ---
        db_contests = self.db_service.get_contests_by_advanced_filter(
            filters={"title": text}, limit=1
        )
        if db_contests:
            db_c = db_contests[0]
            for k in self._get_confidence_keys():
                if not result.get(k) and db_c.get(k):
                    result[k] = db_c[k]
            # Optionally enrich with more fields from DB
            for k in ["district", "office_level", "county_fips", "precinct"]:
                if db_c.get(k):
                    result[k] = db_c[k]

        # --- ML Embedding Similarity (if available) ---
        if self.embedding_model:
            try:
                emb = self.embedding_model.encode([text], show_progress_bar=False)
                result["extra_embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb
            except Exception as e:
                logger.error(f"[ContextBasedPredictor] Embedding failed: {e}")

        # --- Confidence ---
        result["confidence"] = self._estimate_confidence(result)

        # --- Extra ---
        result["extra"] = {
            "raw_text": text,
            "entities": entities,
            "dates": dates,
            "locations": locations,
            "context_version": self.context_service.get_version(),
            "db_enriched": bool(db_contests),
        }
        return result

    async def apredict(self, text: str) -> PredictionResult:
        import asyncio
        await asyncio.sleep(0.01)
        return self.predict(text)

    def _get_confidence_keys(self) -> List[str]:
        # Dynamically build the list from context or librarian
        context_fields = [
            "state", "county", "year", "type_", "office", "party", "candidate",
            "district", "office_level", "county_fips", "precinct"
        ]
        # Extend with any additional fields defined in context library
        extra_fields = self.context_service.get_all("fields")
        for field in extra_fields:
            if field not in context_fields:
                context_fields.append(field)
        return context_fields

    def _estimate_confidence(self, result: PredictionResult) -> float:
        keys = self._get_confidence_keys()
        found = sum(1 for k in keys if result.get(k))
        base = 0.5 + 0.5 * (found / len(keys))
        if result.get("extra", {}).get("db_enriched"):
            base += 0.1
        if "extra_embedding" in result:
            base += 0.05
        return round(min(base, 1.0), 2)

class ContextService:
    """
    Unified, event-driven context/vocab manager for ML/NLP and data integrity.
    Handles vocab export, normalization, enrichment, analytics, versioning, and audit logging.
    """

    def __init__(self, vocab_dir: str):
        self.vocab_dir = vocab_dir
        self.context = load_context_library()
        self.version = self._compute_version()
        self.audit_log_path = os.path.join(self.vocab_dir, AUDIT_LOG)
        self.prompt = UserPrompt()
        self._event_hooks: Dict[str, List[Callable]] = {}
        self._cache = {}

    # --- Context Accessors ---

    def get_all(self, entity_type: str) -> List[str]:
        return self.context.get(entity_type, [])

    def get_all_states(self) -> List[str]:
        return self.get_all("states")

    def get_all_counties(self) -> List[str]:
        return self.get_all("counties")

    def get_all_candidates(self) -> List[str]:
        return self.get_all("candidates")

    def get_all_types(self) -> List[str]:
        return self.get_all("types")

    def get_all_years(self) -> List[str]:
        return self.get_all("years")

    # --- Normalization & Alias Resolution ---

    def normalize_state(self, name: str) -> str:
        return normalize_state_name(name)

    def resolve_county(self, name: str) -> str:
        return resolve_county_alias(name)

    # --- Vocab Export ---

    def export_vocab(self, entity_type: str) -> str:
        """Export vocab file for ML/NLP."""
        items = self.get_all(entity_type)
        path = os.path.join(self.vocab_dir, f"{entity_type}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in sorted(set(items)):
                f.write(f"{item}\n")
        self._log_audit("export_vocab", {"entity_type": entity_type, "count": len(items)})
        return path

    def export_all_vocabs(self) -> Dict[str, str]:
        """Export all vocab files."""
        paths = {}
        for entity_type in self.context.keys():
            paths[entity_type] = self.export_vocab(entity_type)
        return paths

    # --- Versioning & Caching ---

    def _compute_version(self) -> str:
        """Compute a hash/version for the current context."""
        context_json = json.dumps(self.context, sort_keys=True)
        return hashlib.sha256(context_json.encode("utf-8")).hexdigest()

    def get_version(self) -> str:
        return self.version

    def invalidate_cache(self):
        self._cache.clear()
        self.version = self._compute_version()

    # --- Event Hooks (Observer Pattern) ---

    def on(self, event_type: str, handler: Callable):
        """Register an event handler."""
        self._event_hooks.setdefault(event_type, []).append(handler)

    def emit(self, event_type: str, data: Any):
        """Emit an event and call all handlers."""
        for handler in self._event_hooks.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"ContextService event handler error: {e}")
        # Always log the event
        self._log_audit(event_type, data)

    # --- Context Update & Enrichment ---

    def update_from_event(self, event_type: str, data: Any):
        """
        Handle events like new entity, unknown token, context update, etc.
        Enrich context, export vocabs, and log for review.
        """
        logger.info(f"[ContextService] Event: {event_type} | Data: {data}")
        if event_type == "new_entity":
            entity_type = (data or {}).get("entity_type")
            value = (data or {}).get("value")
            if entity_type and value:
                if value not in self.context.setdefault(entity_type, []):
                    safe_append(self.context[entity_type], value, logger, deduplicate=True)
                    self.invalidate_cache()
                    self.export_vocab(entity_type)
                    logger.info(f"[ContextService] Added new {entity_type}: {value}")
        elif event_type == "unknown_token":
            # Log for human review
            self._log_audit("unknown_token", data)
        elif event_type == "context_update":
            # Replace context with new data
            self.context = data
            self.invalidate_cache()
            self.export_all_vocabs()
        # Call event hooks
        self.emit(event_type, data)

    # --- Analytics & Monitoring ---

    def entity_stats(self, entity_type: str) -> Dict[str, int]:
        """Return frequency stats for an entity type."""
        items = self.get_all(entity_type)
        stats = {}
        for item in items:
            stats[item] = stats.get(item, 0) + 1
        return stats

    def unknowns_report(self) -> List[Dict[str, Any]]:
        """Return a list of unknowns/anomalies from audit log."""
        unknowns = []
        if os.path.exists(self.audit_log_path):
            with open(self.audit_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if (entry or {}).get("event") == "unknown_token":
                            unknowns.append(entry)
                    except Exception:
                        continue
        return unknowns

    # --- Human-in-the-Loop CLI Review ---

    def review_unknowns_cli(self):
        """CLI tool for reviewing and resolving unknowns."""
        unknowns = self.unknowns_report()
        if not unknowns:
            logger.info("[ContextService] No unknowns to review.")
            return
        logger.info(f"[ContextService] Reviewing {len(unknowns)} unknowns.")
        for entry in unknowns:
            entity_type = (entry["data"] or {}).get("entity_type")
            value = (entry["data"] or {}).get("value")
            logger.info(f"Unknown {entity_type}: '{value}'")
            if self.prompt.prompt_yes_no(f"Add '{value}' to {entity_type}?", default="n"):
                self.update_from_event("new_entity", {"entity_type": entity_type, "value": value})
                logger.info(f"Added '{value}' to {entity_type}.")
            else:
                logger.info(f"Skipped '{value}'.")

    def review_context_cli(self):
        """CLI tool for reviewing and editing the current context."""
        logger.info("[ContextService] Current context:")
        for k, v in self.context.items():
            logger.info(f"  {k}: {v}")
        if self.prompt.prompt_yes_no("Edit context?", default="n"):
            for k in self.context:
                if self.prompt.prompt_yes_no(f"Edit {k}?", default="n"):
                    new_val = self.prompt.prompt_input(f"Enter new value(s) for {k} (comma-separated):", default=",".join(self.context[k]))
                    self.context[k] = [x.strip() for x in new_val.split(",") if x.strip()]
            self.invalidate_cache()
            self.export_all_vocabs()
            self._log_audit("context_edit", {"context": self.context})
            logger.info("[ContextService] Context updated.")

    # --- Audit Logging ---

    def _log_audit(self, event: str, data: Any):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "data": data,
            "version": self.version
        }
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Provenance/Version Info ---

    def get_audit_log(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.audit_log_path):
            return []
        with open(self.audit_log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_last_update_info(self) -> Optional[Dict[str, Any]]:
        log = self.get_audit_log()
        return log[-1] if log else None

# --- CLI Entrypoint Example ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ContextService CLI")
    parser.add_argument("--vocab-dir", type=str, required=True, help="Path to vocab directory")
    parser.add_argument("--review-unknowns", action="store_true", help="Review and resolve unknowns")
    parser.add_argument("--review-context", action="store_true", help="Review and edit context")
    parser.add_argument("--export-all", action="store_true", help="Export all vocab files")
    parser.add_argument("--stats", type=str, help="Show stats for entity type")
    args = parser.parse_args()

    service = ContextService(args.vocab_dir)

    if args.review_unknowns:
        service.review_unknowns_cli()
    if args.review_context:
        service.review_context_cli()
    if args.export_all:
        service.export_all_vocabs()
        logger.info("Exported all vocab files.")
    if args.stats:
        stats = service.entity_stats(args.stats)
        logger.info(f"Stats for {args.stats}: {stats}")