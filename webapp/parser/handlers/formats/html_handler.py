from __future__ import annotations

import importlib
import os
from typing import Any, Dict, List, Optional, Tuple, cast

import orjson

from ...Context_Integration.Context_Library.constants import KNOWN_COUNTY_TO_PRECINCTS_MAP
from ...state_router import fuzzy_match_handler, get_handler, list_available_handlers
from ...utils.contest_selector import (
    resolve_selection_context,
)
from ...utils.logger_singleton import logger as app_logger
from ...utils.logger_singleton import prompt
from ...utils.shared_logic import normalize_county_name, normalize_state_name, safe_get, safe_parse

# webapp/parser/handlers/formats/html_handler.py
# ---------------------------------------------------------------
# This file is part of the HTML Parser prototype for BallotLens.
# It handles the parsing of HTML pages, routing to appropriate state/county handlers,
# and organizing context for further processing.
# ---------------------------------------------------------------

def parse(
    page: Any,
    coordinator: Any | None = None,
    context: Dict[str, Any] | None = None,
    session_id: Optional[str] = None,
    logger: Any | None = None,
    **kwargs: Any,
) -> Tuple[List[str] | None, List[Dict[str, Any]] | None, str | None, Dict[str, Any]]:
    """
    Generic HTML handler: organizes context, attempts to route to the correct state/county handler,
    and ensures all key election data is transferred to the appropriate downstream handler.
    If no handler is found, uses ML/NLP and user feedback to improve routing, and logs all attempts.
    No extraction is performed here.
    """
    from ...Context_Integration.context_coordinator import ContextCoordinator

    # 1. Organize and enrich context
    html_context = dict(context or {})
    app_logger.debug(f"[HTML Handler] Initial html_context: {html_context}")

    # 2. Use ContextCoordinator for enrichment, NLP, and validation
    active_coordinator = coordinator if getattr(coordinator, "organize_and_enrich", None) else None
    if active_coordinator is None:
        active_coordinator = ContextCoordinator()
    coordinator = active_coordinator  # downstream parse passes this object along
    coordinator.organize_and_enrich(html_context)
    organized: Dict[str, Any] = getattr(coordinator, "organized", {}) or {}
    # Infer and propagate state/county/year if missing
    try:
        st, ct, yr = resolve_selection_context(coordinator=coordinator, context=html_context)
        if st and not html_context.get("state"):
            html_context["state"] = st
        if ct and not html_context.get("county"):
            html_context["county"] = ct
        if yr and not html_context.get("year"):
            html_context["year"] = yr
    except Exception:
        pass
    # 3. Normalize state/county before passing to get_handler
    if "state" in html_context:
        html_context["state"] = normalize_state_name(html_context["state"])
    if "county" in html_context:
        html_context["county"] = normalize_county_name(html_context["county"])

    # 4. Attempt to find handler (first pass)
    handler_info = get_handler(html_context, url=getattr(page, "url", None))
    handler = handler_info["handler"] if isinstance(handler_info, dict) else handler_info
    handler_found = handler and hasattr(handler, "parse") and handler is not parse

    # --- Routing diagnostics ---
    routing_trace = []
    routing_trace.append(f"Initial state: {html_context.get('state')}, county: {html_context.get('county')}")
    attempts: List[Dict[str, Any]] = []
    entities: List[Any] = []
    available_counties: List[str] = []

    # 5. Feedback loop: If handler not found, try ML/NLP and prompt user
    if not handler_found:
        handler_path = prompt.prompt_input("Enter handler path manually (or leave blank to skip): ").strip()
        if handler_path:
            try:
                handler_mod = importlib.import_module(handler_path)
                if callable(handler_mod):
                    handler = handler_mod
                    handler_found = True
                elif hasattr(handler_mod, "parse"):
                    handler = safe_parse(handler_mod)
                    handler_found = True
                else:
                    handler_found = False
                attempts.append({
                    "method": "manual_handler_path",
                    "handler_path": handler_path
                })
                routing_trace.append(f"User specified handler path: {handler_path}")
            except Exception as e:
                app_logger.error(f"[HTML Handler] Failed to import handler from path '{handler_path}': {e}")
                routing_trace.append(f"Failed manual handler import: {handler_path} ({e})")

        # --- ML/NLP: Try to infer state/county from context/entities ---
        state = normalize_state_name(html_context.get("state"))
        county = normalize_county_name(html_context.get("county"))
        url = getattr(page, "url", None) or html_context.get("source_url", "")
        contests = organized.get("contests", [])
        entities = []
        for c in contests:
            entities.extend(safe_get(c, "entities", []))
        try:
            ml_suggestions = coordinator.validate_and_check_integrity()
        except Exception:
            ml_suggestions = {}
        if not isinstance(ml_suggestions, dict):
            ml_suggestions = {}
        suggested_state = normalize_state_name(state or (ml_suggestions.get("integrity_issues") or [{}])[0].get("state"))
        suggested_county = normalize_county_name(county or (ml_suggestions.get("integrity_issues") or [{}])[0].get("county"))
        attempts.append({
            "method": "ml_nlp",
            "suggested_state": suggested_state,
            "suggested_county": suggested_county,
            "entities": entities,
            "url": url
        })
        routing_trace.append(f"ML/NLP suggestions: state={suggested_state}, county={suggested_county}")

        # --- Handler discovery and fuzzy suggestions ---
        available_states = list_available_handlers(level="state")
        available_counties = list_available_handlers(level="county", state=(suggested_state or state or ""))
        app_logger.info(f"[HTML Handler] Available states: {available_states}")
        app_logger.info(f"[HTML Handler] Available counties for state '{suggested_state or state}': {available_counties}")

        # Fuzzy match for county if not found
        if county and county not in available_counties:
            matches = fuzzy_match_handler(county or "", available_counties)
            app_logger.warning(f"[HTML Handler] County '{county}' not found. Closest matches: {matches}")
            routing_trace.append(f"Fuzzy county matches for '{county}': {matches}")

        # --- Context consistency check ---
        if county and (county not in available_counties):
            app_logger.warning(f"[HTML Handler] Detected county '{county}' is not in known counties for state '{suggested_state or state}'.")
            routing_trace.append(f"County '{county}' not in known counties for state '{suggested_state or state}'.")

        # --- Prompt user for manual override ---
        app_logger.info("[HTML Handler] Prompting user for manual state/county selection.")
        max_prompt_attempts = 3
        for attempt in range(max_prompt_attempts):
            try:
                user_state_raw = prompt.prompt_input(
                    f"Enter state (or type 'skip' to stop, blank keeps '{suggested_state or state}'): "
                ).strip()
            except Exception:
                break
            if not user_state_raw:
                user_state_raw = (suggested_state or state) or ""
            if user_state_raw.lower() in {"skip", "cancel"}:
                break
            normalized_state = normalize_state_name(user_state_raw)
            user_state = normalized_state if normalized_state else user_state_raw
            available_states = list_available_handlers(level="state")
            if user_state not in available_states:
                matches = fuzzy_match_handler(user_state or "", available_states)
                app_logger.warning(f"[HTML Handler] State '{user_state}' not found. Closest matches: {matches}")
                if matches:
                    try:
                        confirm = prompt.prompt_input(
                            f"Did you mean '{matches[0]}'? (y/n): "
                        ).strip().lower()
                    except Exception:
                        break
                    proposed = cast(str, matches[0])
                    if confirm == "y" and proposed:
                        normalized_match = normalize_state_name(proposed)
                        user_state = normalized_match if normalized_match else proposed
                    else:
                        continue
                else:
                    app_logger.error(f"[HTML Handler] No valid state handler found for '{user_state}'. Try again.")
                    continue

            available_counties = list_available_handlers(level="county", state=user_state)
            try:
                user_county_raw = prompt.prompt_input(
                    f"Enter county (or type 'skip' to stop, blank keeps '{suggested_county or county}'): "
                ).strip()
            except Exception:
                break
            if not user_county_raw:
                user_county_raw = (suggested_county or county) or ""
            if user_county_raw.lower() in {"skip", "cancel"}:
                break
            normalized_county = normalize_county_name(user_county_raw)
            user_county = normalized_county if normalized_county else user_county_raw
            if user_county not in available_counties:
                known_county_to_precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP
                mapped_county = None
                for county_name, precincts in known_county_to_precincts.items():
                    if user_county in [normalize_county_name(d) for d in precincts]:
                        normalized_mapping = normalize_county_name(county_name)
                        if normalized_mapping:
                            mapped_county = normalized_mapping
                            app_logger.info(f"[HTML Handler] '{user_county}' matched as precincts of county '{county_name}'. Using '{county_name}'.")
                            user_county = normalized_mapping
                            break
                if not mapped_county:
                    matches = fuzzy_match_handler(user_county or "", available_counties)
                    app_logger.warning(f"[HTML Handler] County '{user_county}' not found. Closest matches: {matches}")
                    if matches:
                        try:
                            confirm = prompt.prompt_input(
                                f"Did you mean '{matches[0]}'? (y/n): "
                            ).strip().lower()
                        except Exception:
                            break
                        proposed = cast(str, matches[0])
                        if confirm == "y" and proposed:
                            normalized_match = normalize_county_name(proposed)
                            user_county = normalized_match if normalized_match else proposed
                        else:
                            continue
                    else:
                        app_logger.error(f"[HTML Handler] No valid county handler found for '{user_county}'. Try again.")
                        continue

            html_context["state"] = user_state
            html_context["county"] = user_county
            handler_info = get_handler(html_context, url=url)
            handler = handler_info["handler"] if isinstance(handler_info, dict) else handler_info
            handler_found = handler and hasattr(handler, "parse") and handler is not parse
            attempts.append({
                "method": "manual_prompt",
                "user_state": user_state,
                "user_county": user_county
            })
            routing_trace.append(f"User override: state={user_state}, county={user_county}")
            if handler_found:
                break

            routing_trace.append("Manual selection did not resolve to a handler.")
            try:
                retry = prompt.prompt_input(
                    "No handler found for that selection. Try another state/county? (y/n): ",
                    default="n",
                ).strip().lower()
            except Exception:
                break
            if retry not in ("y", "yes"):
                break

        # Optionally allow user to specify handler path directly
        if not handler_found:
            handler_path = prompt.prompt_input("Enter handler path manually (or leave blank to skip): ").strip()
            if handler_path:
                try:
                    handler_mod = importlib.import_module(handler_path)
                    handler = getattr(handler_mod, "parse", None)
                    handler_found = handler is not None
                    attempts.append({
                        "method": "manual_handler_path",
                        "handler_path": handler_path
                    })
                    routing_trace.append(f"User specified handler path: {handler_path}")
                except Exception as e:
                    app_logger.error(f"[HTML Handler] Failed to import handler from path '{handler_path}': {e}")
                    routing_trace.append(f"Failed manual handler import: {handler_path} ({e})")

    # 6. If handler found after feedback, route and return
    if handler_found:
        app_logger.info(f"[HTML Handler] Routing to state/county handler: {getattr(handler, '__name__', str(handler))}")
        app_logger.info(f"[HTML Handler] Routing trace: {routing_trace}")
        # Pass enriched context and coordinator downstream
        return safe_parse(
            handler,
            page,
            coordinator,
            html_context,
            session_id=session_id,
            logger=logger,
            **kwargs
        )

    # 7. If still not found, log all attempts and provide actionable error
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "html_handler_routing_failures.jsonl")
    with open(log_path, "ab") as f:
        f.write(
            orjson.dumps(
                {
                    "url": getattr(page, "url", None) or html_context.get("source_url", ""),
                    "context": html_context,
                    "attempts": attempts,
                    "routing_trace": routing_trace
                }
            ) + b"\n"
        )

    # Offer to export context for manual review
    export = prompt.prompt_input("Routing failed. Export organized context for debugging? (y/n): ").strip().lower()
    if export == "y":
        export_path = os.path.join(log_dir, "html_handler_failed_context.json")
        with open(export_path, "wb") as ef:
            ef.write(orjson.dumps(html_context, option=orjson.OPT_INDENT_2))
        (logger or app_logger).info(f"[HTML Handler] Context exported to {export_path}")

    app_logger.error("[HTML Handler] No suitable handler could be found after all attempts. Routing failed.")
    app_logger.info(f"[HTML Handler] Routing trace: {routing_trace}")
    app_logger.info(f"[HTML Handler] Entities used for routing: {entities}")
    app_logger.info(f"[HTML Handler] Available handlers for state '{html_context.get('state')}': {available_counties}")

    return None, None, None, {
        "skipped": True,
        "reason": "No suitable handler found after ML/NLP/user feedback.",
        "context": html_context,
        "attempts": attempts,
        "routing_trace": routing_trace,
        "available_handlers": available_counties
    }