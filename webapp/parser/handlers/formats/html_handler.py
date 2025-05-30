def parse(page, coordinator=None, context=None, non_interactive=False, **kwargs):
    """
    Generic HTML handler: organizes context, delegates to state/county handler if available,
    and ensures all key election data is extracted and categorized for downstream use.
    Returns headers, data, contest_title, metadata.
    """
    from ...Context_Integration.context_coordinator import ContextCoordinator
    from ...state_router import get_handler
    from ...utils.contest_selector import select_contest
    from ...utils.table_builder import build_dynamic_table, rescan_and_verify
    from ...utils.shared_logic import  find_and_click_toggle
    from ...utils.logger_instance import logger

    # 1. Scan and enrich HTML context
    html_context = context or {}
    if context:
        html_context.update(context)
    logger.debug(f"[HTML Handler] Initial html_context: {html_context}")

    # 2. Use ContextCoordinator for enrichment, NLP, and validation
    if coordinator is None:
        coordinator = ContextCoordinator()
    coordinator.organize_and_enrich(html_context)
    organized = coordinator.organized or {}

    # 3. Route to state/county handler if available
    handler = get_handler(html_context, url=getattr(page, "url", None))
    if handler and hasattr(handler, "parse") and handler is not parse:
        logger.info(f"[HTML Handler] Routing to state/county handler: {handler.__name__}")
        # Pass enriched context and coordinator downstream
        return handler.parse(page, coordinator, html_context, non_interactive=non_interactive, **kwargs)

    # 4. Contest selection (Elector Race, Election Type, Year, Districts)
    contests = organized.get("contests", [])
    contest_title = html_context.get("selected_race")
    if not contest_title:
        selected = select_contest(
            coordinator,
            state=html_context.get("state"),
            county=html_context.get("county"),
            year=html_context.get("year"),
            non_interactive=non_interactive
        )
        if not selected:
            logger.warning("[HTML Handler] No contest selected. Skipping.")
            return None, None, None, {"skipped": True}
        # Support multiple contests (e.g., multiple years/types)
        if isinstance(selected, list) and selected:
            contest_title = selected[0].get("title", "") if isinstance(selected[0], dict) else selected[0]
        else:
            contest_title = selected
        html_context["selected_race"] = contest_title

    # 5. Find and click the correct toggle/button for the contest (using NLP/spaCy if needed)
    # Use coordinator/library to get best button for this contest
    button = coordinator.get_best_button(contest_title, keywords=["View Results", "Vote Method", "Show Results"])
    if button:
        clicked = find_and_click_toggle(
            page,
            coordinator,
            container=None,
            handler_keywords=[button.get("label")],
            logger=logger,
            verbose=True,
        )
        if not clicked:
            logger.warning("[HTML Handler] Could not click contest toggle/button automatically.")
    else:
        logger.warning("[HTML Handler] No suitable button found for contest.")

    # 6. Extract contest panel and tables (districts, candidates, results)
    # Try to find a contest panel for the selected contest
    contest_panel = None
    if hasattr(handler, "extract_contest_panel"):
        contest_panel = handler.extract_contest_panel(page, contest_title)
    if not contest_panel:
        # Fallback: try generic panel extraction
        contest_panel = page.locator("div, section, article").filter(has_text=contest_title)
        if contest_panel.count() > 0:
            contest_panel = contest_panel.first
        else:
            contest_panel = None

    # Extract tables (districts, candidates, results)
    headers, data = [], []
    if contest_panel:
        # Try to extract precinct tables (districts)
        if hasattr(handler, "extract_precinct_tables"):
            precinct_tables = handler.extract_precinct_tables(contest_panel)
        else:
            precinct_tables = []
        for precinct_name, table in precinct_tables:
            table_headers, rows = build_dynamic_table(table)
            for row in rows:
                row["District"] = precinct_name
                data.append(row)
            if table_headers:
                headers = table_headers
    else:
        # Fallback: extract first table on the page
        table = page.locator("table")
        if table.count() > 0:
            headers, data = build_dynamic_table(table.first)

    # 7. Organize output for wide format and downstream use
    # Ensure all required fields are present and categorized
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    # Wide format: District, %Reported, Candidate, Party, Ballot Type, Votes, etc.
    preferred_order = [
        "District", "% Precincts Reporting", "Candidate", "Party", "Ballot Type", "Votes"
    ]
    headers = [h for h in preferred_order if h in all_keys] + [h for h in all_keys if h not in preferred_order]
    # Add grand totals row if data exists
    if data:
        data.append(rescan_and_verify(data))

    # 8. Build metadata for output
    contest_info = next((c for c in contests if c.get("title") == contest_title), {})
    metadata = {
        "state": html_context.get("state", contest_info.get("state", "Unknown")),
        "county": html_context.get("county", contest_info.get("county", "Unknown")),
        "year": html_context.get("year", contest_info.get("year", "Unknown")),
        "election_type": contest_info.get("type", html_context.get("election_type", "Unknown")),
        "race": contest_title or contest_info.get("title", "Unknown"),
        "districts": [row.get("District") for row in data if "District" in row],
        "candidates": list({row.get("Candidate") for row in data if "Candidate" in row}),
        "parties": list({row.get("Party") for row in data if "Party" in row}),
        "ballot_types": list({row.get("Ballot Type") for row in data if "Ballot Type" in row}),
        "source": getattr(page, "url", html_context.get("source_url", "Unknown")),
        "handler": "html_handler"
    }

    return headers, data, contest_title, metadata