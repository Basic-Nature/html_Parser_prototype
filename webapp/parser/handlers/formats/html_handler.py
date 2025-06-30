def parse(page, coordinator=None, context=None, non_interactive=False, **kwargs):
    """
    Generic HTML handler: organizes context, attempts to route to the correct state/county handler,
    and ensures all key election data is transferred to the appropriate downstream handler.
    If no handler is found, uses ML/NLP and user feedback to improve routing, and logs all attempts.
    No extraction is performed here.
    """
    from ...Context_Integration.context_coordinator import ContextCoordinator
    from ...state_router import get_handler, list_available_handlers, fuzzy_match_handler
    from ...utils.shared_logic import normalize_state_name, normalize_county_name
    from ...utils.shared_logger import log_info, log_debug, log_warning, log_error
    from ...utils.user_prompt import prompt_user_input
    import orjson
    import os
    import importlib
    from ...bots.librarian import KNOWN_COUNTY_TO_PRECINCTS_MAP
    # 1. Organize and enrich context
    html_context = context or {}
    if context:
        html_context.update(context)
    log_debug(f"[HTML Handler] Initial html_context: {html_context}")

    # 2. Use ContextCoordinator for enrichment, NLP, and validation
    if coordinator is None:
        coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(html_context)
    organized = coordinator.organized or {}

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
    attempts = []

    # 5. Feedback loop: If handler not found, try ML/NLP and prompt user
    if not handler_found:
        handler_path = prompt_user_input("Enter handler path manually (or leave blank to skip): ").strip()
        if handler_path:
            try:
                handler_mod = importlib.import_module(handler_path)
                if callable(handler_mod):
                    handler = handler_mod
                    handler_found = True
                elif hasattr(handler_mod, "parse"):
                    handler = handler_mod.parse
                    handler_found = True
                else:
                    handler_found = False
                attempts.append({
                    "method": "manual_handler_path",
                    "handler_path": handler_path
                })
                routing_trace.append(f"User specified handler path: {handler_path}")
            except Exception as e:
                log_error(f"[HTML Handler] Failed to import handler from path '{handler_path}': {e}")
                routing_trace.append(f"Failed manual handler import: {handler_path} ({e})")

        # --- ML/NLP: Try to infer state/county from context/entities ---
        state = normalize_state_name(html_context.get("state"))
        county = normalize_county_name(html_context.get("county"))
        url = getattr(page, "url", None) or html_context.get("source_url", "")
        contests = organized.get("contests", [])
        entities = []
        for c in contests:
            entities.extend(c.get("entities", []))
        ml_suggestions = coordinator.validate_and_check_integrity()
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
        available_counties = list_available_handlers(level="county", state=suggested_state or state)
        log_info(f"[HTML Handler] Available states: {available_states}")
        log_info(f"[HTML Handler] Available counties for state '{suggested_state or state}': {available_counties}")

        # Fuzzy match for county if not found
        if county and county not in available_counties:
            matches = fuzzy_match_handler(county, available_counties)
            log_warning(f"[HTML Handler] County '{county}' not found. Closest matches: {matches}")
            routing_trace.append(f"Fuzzy county matches for '{county}': {matches}")

        # --- Context consistency check ---
        if county and (county not in available_counties):
            log_warning(f"[HTML Handler] Detected county '{county}' is not in known counties for state '{suggested_state or state}'.")
            routing_trace.append(f"County '{county}' not in known counties for state '{suggested_state or state}'.")

        # --- Prompt user for manual override ---
        if not non_interactive:
            log_info("[HTML Handler] Prompting user for manual state/county selection.")
            while True:
                user_state = prompt_user_input(
                    f"Enter state (or leave blank to keep '{suggested_state or state}'): "
                ).strip() or (suggested_state or state)
                user_state = normalize_state_name(user_state)
                available_states = list_available_handlers(level="state")
                if user_state not in available_states:
                    matches = fuzzy_match_handler(user_state, available_states)
                    log_warning(f"[HTML Handler] State '{user_state}' not found. Closest matches: {matches}")
                    if matches:
                        confirm = prompt_user_input(
                            f"Did you mean '{matches[0]}'? (y/n): "
                        ).strip().lower()
                        if confirm == "y":
                            user_state = matches[0]
                        else:
                            continue
                    else:
                        log_error(f"[HTML Handler] No valid state handler found for '{user_state}'. Try again.")
                        continue

                available_counties = list_available_handlers(level="county", state=user_state)
                user_county = prompt_user_input(
                    f"Enter county (or leave blank to keep '{suggested_county or county}'): "
                ).strip() or (suggested_county or county)
                user_county = normalize_county_name(user_county)
                if user_county not in available_counties:
                    # Check if county is a precincts mapped in context library
                    known_county_to_precincts = KNOWN_COUNTY_TO_PRECINCTS_MAP
                    mapped_county = None
                    for county_name, precincts in known_county_to_precincts.items():
                        if user_county in [normalize_county_name(d) for d in precincts]:
                            mapped_county = normalize_county_name(county_name)
                            log_info(f"[HTML Handler] '{user_county}' matched as precincts of county '{county_name}'. Using '{county_name}'.")
                            user_county = mapped_county
                            break
                    if not mapped_county:
                        matches = fuzzy_match_handler(user_county, available_counties)
                        log_warning(f"[HTML Handler] County '{user_county}' not found. Closest matches: {matches}")
                        if matches:
                            confirm = prompt_user_input(
                                f"Did you mean '{matches[0]}'? (y/n): "
                            ).strip().lower()
                            if confirm == "y":
                                user_county = matches[0]
                            else:
                                continue
                        else:
                            log_error(f"[HTML Handler] No valid county handler found for '{user_county}'. Try again.")
                            continue

                # If we get here, both state and county are valid
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

            # Optionally allow user to specify handler path directly
            if not handler_found:
                handler_path = prompt_user_input("Enter handler path manually (or leave blank to skip): ").strip()
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
                        log_error(f"[HTML Handler] Failed to import handler from path '{handler_path}': {e}")
                        routing_trace.append(f"Failed manual handler import: {handler_path} ({e})")

    # 6. If handler found after feedback, route and return
    if handler_found:
        log_info(f"[HTML Handler] Routing to state/county handler: {getattr(handler, '__name__', str(handler))}")
        log_info(f"[HTML Handler] Routing trace: {routing_trace}")
        # Pass enriched context and coordinator downstream
        return handler.parse(page, coordinator, html_context, non_interactive=non_interactive, **kwargs)

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
    if not non_interactive:
        export = prompt_user_input("Routing failed. Export organized context for debugging? (y/n): ").strip().lower()
        if export == "y":
            export_path = os.path.join(log_dir, "html_handler_failed_context.json")
            with open(export_path, "wb") as ef:
                ef.write(orjson.dumps(html_context, option=orjson.OPT_INDENT_2))
            log_info(f"[HTML Handler] Context exported to {export_path}")

    log_error("[HTML Handler] No suitable handler could be found after all attempts. Routing failed.")
    log_info(f"[HTML Handler] Routing trace: {routing_trace}")
    log_info(f"[HTML Handler] Entities used for routing: {entities}")
    log_info(f"[HTML Handler] Available handlers for state '{html_context.get('state')}': {available_counties}")

    return None, None, None, {
        "skipped": True,
        "reason": "No suitable handler found after ML/NLP/user feedback.",
        "context": html_context,
        "attempts": attempts,
        "routing_trace": routing_trace,
        "available_handlers": available_counties
    }