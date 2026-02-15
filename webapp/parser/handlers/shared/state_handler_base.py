"""
State Handler Base Class

Provides common workflow hooks and utilities for state-specific handlers.
Designed to reduce code duplication and standardize handler implementation.

Usage:
    from webapp.parser.handlers.shared.state_handler_base import StateHandlerBase
    
    class CaliforniaHandler(StateHandlerBase):
        STATE_NAME = "California"
        STATE_CODE = "CA"
        
        def extract_tables(self, page, html_context, coordinator, session_id):
            # Custom extraction logic for California
            return headers, data_rows
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from webapp.parser.Context_Integration.librarian import clean_for_json
from webapp.parser.handlers.shared.parity_hooks import (
    attach_parity_note_to_metadata,
    extract_router_parity_note,
)
from webapp.parser.utils.contest_selector import select_contest_auto_first
from webapp.parser.utils.html_scanner import scan_html_for_context
from webapp.parser.utils.logger_singleton import logger
from webapp.parser.utils.retry_utils import retry_with_snapshot
from webapp.parser.utils.table_core import robust_table_extraction


class StateHandlerBase(ABC):
    """
    Abstract base class for state-specific election data handlers.
    
    Provides:
    - Standard parse() workflow with customizable hooks
    - Context scanning and enrichment
    - Contest detection and selection
    - Table extraction delegation
    - Metadata assembly
    - Logging and error handling
    
    Subclasses must implement:
    - STATE_NAME: Full state name (e.g., "California")
    - STATE_CODE: 2-letter code (e.g., "CA")
    - extract_tables(): Custom table extraction logic
    
    Subclasses may override:
    - pre_scan_hook(): Run before HTML scanning
    - post_scan_hook(): Run after HTML scanning
    - pre_extraction_hook(): Run before table extraction
    - post_extraction_hook(): Run after table extraction
    - should_use_fallback(): Determine if fallback to generic handler needed
    """
    
    # Subclasses must define these
    STATE_NAME: str = None
    STATE_CODE: str = None
    
    def __init__(self):
        """Initialize handler with state configuration."""
        if not self.STATE_NAME or not self.STATE_CODE:
            raise ValueError(f"{self.__class__.__name__} must define STATE_NAME and STATE_CODE")
        
        # Enable auto-retry by default (can be disabled by subclasses)
        self.enable_auto_retry = True
        self.max_retry_attempts = 3
        self.retry_backoff = 2.0
    
    def parse(
        self,
        page: Any = None,
        html_context: Dict[str, Any] | None = None,
        coordinator: Any = None,
        context: Dict[str, Any] | None = None,
        session_id: str | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]] | None:
        """
        Main parse entry point. Follows standard workflow with optional auto-retry.
        
        If enable_auto_retry is True (default), automatically retries on failure
        with exponential backoff and snapshot mode on final attempt.
        """
        if self.enable_auto_retry:
            # Wrap with retry decorator
            @retry_with_snapshot(
                max_attempts=self.max_retry_attempts,
                backoff=self.retry_backoff,
                snapshot_on_final_fail=True,
            )
            def parse_with_retry():
                return self._parse_internal(page, html_context, coordinator, context, session_id, **kwargs)
            
            return parse_with_retry()
        else:
            # Direct execution without retry
            return self._parse_internal(page, html_context, coordinator, context, session_id, **kwargs)
    
    def _parse_internal(
        self,
        page: Any,
        html_context: Dict[str, Any] | None,
        coordinator: Any,
        context: Dict[str, Any] | None,
        session_id: str | None,
        **kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]], str, Dict[str, Any]] | None:
        """
        Internal parse implementation. Called by parse() with or without retry wrapper.
        
        Workflow:
        1. Pre-scan hook
        2. Scan HTML for context and contests
        3. Post-scan hook
        4. Select contest
        5. Pre-extraction hook
        6. Extract tables
        7. Post-extraction hook
        8. Build metadata
        9. Return (headers, data_rows, contest_title, metadata)
        """
        try:
            html_context = html_context or context or {}
            session_id = session_id or html_context.get("session_id")
            
            logger.info(f"[bold cyan][{self.STATE_NAME} Handler] Starting parse workflow...[/bold cyan]")
            
            # Extract parity notes for CLI/Web tracking
            parity_note = extract_router_parity_note(html_context)
            
            # Hook: pre-scan customization
            self.pre_scan_hook(page, html_context, coordinator, session_id, **kwargs)
            
            # Check if we should fallback to generic handler
            if self.should_use_fallback(page, html_context):
                logger.info(f"[{self.STATE_NAME}] Delegating to fallback handler")
                return self._fallback_parse(page, html_context, coordinator, session_id, parity_note, **kwargs)
            
            # Scan HTML for context
            context_result = self.scan_for_contests(page, html_context, coordinator, session_id, **kwargs)
            
            # Ensure state/county are set
            context_result = self._ensure_location_fields(context_result, html_context)
            
            # Hook: post-scan customization
            self.post_scan_hook(context_result, html_context, coordinator, session_id, **kwargs)
            
            # Select contest
            selected_contest = self.select_contest(context_result, html_context, coordinator, session_id)
            
            if not selected_contest:
                logger.warning(f"[{self.STATE_NAME}] No contest selected")
                return None, None, None, {"skipped": True, "state": self.STATE_CODE}
            
            # Hook: pre-extraction customization
            self.pre_extraction_hook(selected_contest, page, html_context, coordinator, session_id, **kwargs)
            
            # Extract tables (custom logic in subclass)
            headers, data_rows = self.extract_tables(
                page, selected_contest, html_context, coordinator, session_id, **kwargs
            )
            
            # Hook: post-extraction customization
            headers, data_rows = self.post_extraction_hook(
                headers, data_rows, selected_contest, html_context, coordinator, session_id, **kwargs
            )
            
            # Build metadata
            metadata = self.build_metadata(
                selected_contest, headers, data_rows, context_result, html_context, session_id
            )
            
            # Attach parity note
            metadata = attach_parity_note_to_metadata(metadata, parity_note)
            
            # Log extraction attempt
            self.log_extraction_attempt(selected_contest, headers, data_rows, metadata, success=True)
            
            contest_title = selected_contest.get("title", "Unknown Contest")
            logger.info(f"[green][{self.STATE_NAME}] Parse completed: {len(data_rows)} rows extracted[/green]")
            
            return headers, data_rows, contest_title, metadata
            
        except Exception as e:
            logger.error(f"[red][{self.STATE_NAME}] Parse failed: {e}[/red]")
            self.log_extraction_attempt({}, [], [], {}, success=False, error=str(e))
            return None, None, None, {"error": str(e), "state": self.STATE_CODE}
    
    def scan_for_contests(
        self,
        page: Any,
        html_context: Dict[str, Any],
        coordinator: Any,
        session_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Scan HTML for contests and context using html_scanner.
        
        Returns context_result with contests, segments, buttons, etc.
        """
        from webapp.parser.Context_Integration.context_coordinator import ContextCoordinator
        
        if coordinator is None:
            coordinator = ContextCoordinator()
        
        context_result = scan_html_for_context(
            target_url=getattr(page, "url", None) if page else html_context.get("url"),
            page=page,
            coordinator=coordinator,
            session_id=session_id,
            allow_duplicates=getattr(coordinator, "allow_duplicates", False),
            context_cache={},
            debug=html_context.get("debug", False),
            **kwargs,
        )
        
        # Clean and enrich
        context_result = clean_for_json(context_result)
        enriched = coordinator.organize_and_enrich(context_result)
        
        # ML: Predict missing fields (state/county/year) using ContestFieldClassifier
        if enriched:
            enriched = coordinator.predict_missing_fields(enriched)
        
        # Merge enriched data back into context_result
        if enriched:
            context_result.update(enriched)
        
        return context_result
    
    def select_contest(
        self,
        context_result: Dict[str, Any],
        html_context: Dict[str, Any],
        coordinator: Any,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Select contest from detected contests using contest_selector.
        
        Returns single contest dict or None if no selection made.
        """
        contests = context_result.get("contests", [])
        
        if not contests:
            logger.warning(f"[{self.STATE_NAME}] No contests detected in HTML")
            return None
        
        context_for_selector = {
            "state": context_result.get("state", self.STATE_CODE),
            "county": context_result.get("county"),
            "year": context_result.get("year"),
            "contests": contests,
            "session_id": session_id,
        }
        
        selected = select_contest_auto_first(
            coordinator=coordinator,
            context=context_for_selector,
            session_id=session_id,
            allow_multiple=False,
            force_interactive=html_context.get("force_interactive", False),
        )
        
        # Handle list return (if multiple contests selected)
        if isinstance(selected, list) and selected:
            selected = selected[0]  # Take first for now
        
        return selected if isinstance(selected, dict) else None
    
    @abstractmethod
    def extract_tables(
        self,
        page: Any,
        contest: Dict[str, Any],
        html_context: Dict[str, Any],
        coordinator: Any,
        session_id: str,
        **kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Extract table data for the selected contest.
        
        This is the main customization point for state-specific logic.
        
        Args:
            page: Playwright Page object (if available)
            contest: Selected contest dict with title, state, county, year
            html_context: Full context dict
            coordinator: ContextCoordinator instance
            session_id: Current session ID
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (headers, data_rows)
            - headers: List of column names
            - data_rows: List of dicts, each representing one row
        
        Example implementation:
            def extract_tables(self, page, contest, html_context, coordinator, session_id, **kwargs):
                # Option 1: Delegate to robust_table_extraction
                result = robust_table_extraction(
                    page=page,
                    coordinator=coordinator,
                    html_context=html_context,
                    session_id=session_id,
                )
                return result.get("headers", []), result.get("data", [])
                
                # Option 2: Custom extraction logic
                from webapp.parser.utils.table_builder import build_dynamic_table
                tables = page.query_selector_all("table.results")
                headers, data = build_dynamic_table(tables[0], coordinator)
                return headers, data
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement extract_tables()")
    
    def build_metadata(
        self,
        contest: Dict[str, Any],
        headers: List[str],
        data_rows: List[Dict[str, Any]],
        context_result: Dict[str, Any],
        html_context: Dict[str, Any],
        session_id: str,
    ) -> Dict[str, Any]:
        """
        Build metadata dict for output.
        
        Override to add state-specific metadata fields.
        """
        return {
            "state": contest.get("state", self.STATE_CODE),
            "county": contest.get("county"),
            "year": contest.get("year"),
            "contest_title": contest.get("title"),
            "source_url": html_context.get("url") or context_result.get("url"),
            "session_id": session_id,
            "handler": self.__class__.__name__,
            "row_count": len(data_rows),
            "column_count": len(headers),
        }
    
    def log_extraction_attempt(
        self,
        contest: Dict[str, Any],
        headers: List[str],
        data_rows: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        success: bool,
        error: str = None,
    ):
        """
        Log extraction attempt for learning and debugging.
        
        Override to customize logging behavior.
        """
        log_entry = {
            "handler": self.__class__.__name__,
            "state": self.STATE_CODE,
            "contest": contest.get("title") if contest else None,
            "success": success,
            "row_count": len(data_rows) if success else 0,
            "session_id": metadata.get("session_id") if metadata else None,
        }
        
        if error:
            log_entry["error"] = error
        
        if success:
            logger.debug(f"[{self.STATE_NAME}] Extraction logged: {log_entry}")
        else:
            logger.error(f"[{self.STATE_NAME}] Extraction failed: {log_entry}")
    
    # === Hooks for customization ===
    
    def pre_scan_hook(self, page, html_context, coordinator, session_id, **kwargs):
        """Run before HTML scanning. Override for custom setup."""
        pass
    
    def post_scan_hook(self, context_result, html_context, coordinator, session_id, **kwargs):
        """Run after HTML scanning. Override for custom enrichment."""
        pass
    
    def pre_extraction_hook(self, contest, page, html_context, coordinator, session_id, **kwargs):
        """Run before table extraction. Override for custom navigation/setup."""
        pass
    
    def post_extraction_hook(self, headers, data_rows, contest, html_context, coordinator, session_id, **kwargs):
        """Run after table extraction. Override for custom cleanup/transformation."""
        return headers, data_rows
    
    def should_use_fallback(self, page, html_context) -> bool:
        """
        Determine if handler should fallback to generic HTML extraction.
        
        Override to add conditions (e.g., URL pattern mismatch, missing elements).
        """
        return False
    
    # === Private helpers ===
    
    def _ensure_location_fields(self, context_result: Dict[str, Any], html_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure state/county fields are populated."""
        if not context_result.get("state"):
            context_result["state"] = self.STATE_CODE
        
        # If county provided in html_context, use it
        if not context_result.get("county") and html_context.get("county"):
            context_result["county"] = html_context["county"]
        
        # Propagate to all contests
        for contest in context_result.get("contests", []):
            contest.setdefault("state", self.STATE_CODE)
            if context_result.get("county"):
                contest.setdefault("county", context_result["county"])
            if context_result.get("year"):
                contest.setdefault("year", context_result["year"])
        
        return context_result
    
    def _fallback_parse(self, page, html_context, coordinator, session_id, parity_note, **kwargs):
        """Delegate to generic HTML fallback handler."""
        from webapp.parser.handlers.formats.html_dynamic_fallback import parse as dynamic_parse
        
        result = dynamic_parse(
            page=page,
            coordinator=coordinator,
            context=html_context,
            session_id=session_id,
            **kwargs,
        )
        
        if result and isinstance(result, tuple) and len(result) == 4:
            headers, rows, contest, metadata = result
            metadata = attach_parity_note_to_metadata(metadata, parity_note)
            metadata["state"] = self.STATE_CODE
            metadata["fallback_used"] = True
            return headers, rows, contest, metadata
        
        return result


class SimpleTableHandler(StateHandlerBase):
    """
    Convenience subclass for states with simple table extraction.
    
    Delegates extract_tables() to robust_table_extraction automatically.
    """
    
    def extract_tables(self, page, contest, html_context, coordinator, session_id, **kwargs):
        """Delegate to robust_table_extraction strategy pipeline."""
        result = robust_table_extraction(
            page=page,
            coordinator=coordinator,
            html_context={**html_context, "selected_contest": contest},
            session_id=session_id,
            **kwargs,
        )
        
        headers = result.get("headers", [])
        data_rows = result.get("data", [])
        
        return headers, data_rows
