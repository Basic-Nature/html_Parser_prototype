import os
import json
import hashlib
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Set
from ..bots.librarian import (
    load_context_library,
    resolve_county_alias,
    # Add more normalization/alias utilities as needed
)
from ..utils.shared_logic import normalize_state_name
from ..utils.shared_logger import log_info, log_error
from ..utils.user_prompt import UserPrompt

AUDIT_LOG = "context_audit_log.jsonl"

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
                log_error(f"ContextService event handler error: {e}")
        # Always log the event
        self._log_audit(event_type, data)

    # --- Context Update & Enrichment ---

    def update_from_event(self, event_type: str, data: Any):
        """
        Handle events like new entity, unknown token, context update, etc.
        Enrich context, export vocabs, and log for review.
        """
        log_info(f"[ContextService] Event: {event_type} | Data: {data}")
        if event_type == "new_entity":
            entity_type = data.get("entity_type")
            value = data.get("value")
            if entity_type and value:
                if value not in self.context.setdefault(entity_type, []):
                    self.context[entity_type].append(value)
                    self.invalidate_cache()
                    self.export_vocab(entity_type)
                    log_info(f"[ContextService] Added new {entity_type}: {value}")
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
                        if entry.get("event") == "unknown_token":
                            unknowns.append(entry)
                    except Exception:
                        continue
        return unknowns

    # --- Human-in-the-Loop CLI Review ---

    def review_unknowns_cli(self):
        """CLI tool for reviewing and resolving unknowns."""
        unknowns = self.unknowns_report()
        if not unknowns:
            log_info("[ContextService] No unknowns to review.")
            return
        log_info(f"[ContextService] Reviewing {len(unknowns)} unknowns.")
        for entry in unknowns:
            entity_type = entry["data"].get("entity_type")
            value = entry["data"].get("value")
            log_info(f"Unknown {entity_type}: '{value}'")
            if self.prompt.prompt_yes_no(f"Add '{value}' to {entity_type}?", default="n"):
                self.update_from_event("new_entity", {"entity_type": entity_type, "value": value})
                log_info(f"Added '{value}' to {entity_type}.")
            else:
                log_info(f"Skipped '{value}'.")

    def review_context_cli(self):
        """CLI tool for reviewing and editing the current context."""
        log_info("[ContextService] Current context:")
        for k, v in self.context.items():
            log_info(f"  {k}: {v}")
        if self.prompt.prompt_yes_no("Edit context?", default="n"):
            for k in self.context:
                if self.prompt.prompt_yes_no(f"Edit {k}?", default="n"):
                    new_val = self.prompt.prompt_input(f"Enter new value(s) for {k} (comma-separated):", default=",".join(self.context[k]))
                    self.context[k] = [x.strip() for x in new_val.split(",") if x.strip()]
            self.invalidate_cache()
            self.export_all_vocabs()
            self._log_audit("context_edit", {"context": self.context})
            log_info("[ContextService] Context updated.")

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
        log_info("Exported all vocab files.")
    if args.stats:
        stats = service.entity_stats(args.stats)
        log_info(f"Stats for {args.stats}: {stats}")