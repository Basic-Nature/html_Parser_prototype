"""
Retry Utilities with Snapshot Mode

Provides decorators and utilities for automatic retry logic with escalation
to snapshot mode on final failure.

Features:
- Exponential backoff between retries
- Automatic HTML snapshot capture on final failure
- Logging of failure patterns for learning
- Integration with existing snapshot_mode_pipeline

Usage:
    from webapp.parser.utils.retry_utils import retry_with_snapshot
    
    @retry_with_snapshot(max_attempts=3, backoff=2.0)
    def extract_data(page, html_context, **kwargs):
        # Your extraction logic here
        return result
"""
import functools
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

from .logger_singleton import logger


def retry_with_snapshot(
    max_attempts: int = 3,
    backoff: float = 2.0,
    snapshot_on_final_fail: bool = True,
    exceptions: Tuple[type, ...] = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff and snapshot mode on final failure.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        backoff: Multiplier for exponential backoff delay in seconds (default: 2.0)
        snapshot_on_final_fail: Save HTML snapshot if all retries fail (default: True)
        exceptions: Tuple of exception types to catch and retry (default: all exceptions)
    
    Behavior:
        Attempt 1: Normal execution
        Attempt 2: Wait backoff seconds, retry with fresh context
        Attempt 3: Wait backoff^2 seconds, retry with snapshot_mode=True
        Final fail: Save HTML snapshot and log failure
    
    Example:
        @retry_with_snapshot(max_attempts=3, backoff=2.0)
        def parse_handler(page, html_context, coordinator, session_id, **kwargs):
            return handler.parse(page, html_context, coordinator, session_id, **kwargs)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # Enable snapshot mode on final attempt
                    if attempt == max_attempts and snapshot_on_final_fail:
                        logger.info(f"[retry] Attempt {attempt}/{max_attempts} with snapshot mode enabled")
                        # Find html_context in args/kwargs and set snapshot_mode
                        html_context = _get_html_context(args, kwargs)
                        if html_context is not None:
                            html_context["snapshot_mode"] = True
                    else:
                        logger.info(f"[retry] Attempt {attempt}/{max_attempts}")
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Success - log if this was a retry
                    if attempt > 1:
                        logger.info(f"[green][retry] Succeeded on attempt {attempt}/{max_attempts}[/green]")
                    
                    return result
                
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"[yellow][retry] Attempt {attempt}/{max_attempts} failed: {e}[/yellow]")
                    
                    # If not last attempt, wait and retry
                    if attempt < max_attempts:
                        delay = backoff ** (attempt - 1)  # Exponential: 1, 2, 4 seconds for backoff=2
                        logger.info(f"[retry] Waiting {delay}s before retry...")
                        time.sleep(delay)
                    else:
                        # Last attempt failed
                        logger.error(f"[red][retry] All {max_attempts} attempts failed[/red]")
                        
                        # Save snapshot if enabled
                        if snapshot_on_final_fail:
                            _save_failure_snapshot(args, kwargs, last_exception)
                        
                        # Log failure pattern for learning
                        _log_extraction_failure(args, kwargs, last_exception, max_attempts)
                        
                        # Re-raise the exception
                        raise
            
            # Should not reach here, but handle defensively
            if last_exception:
                raise last_exception
            
            return None
        
        return wrapper
    return decorator


def _get_html_context(args: tuple, kwargs: dict) -> Optional[dict]:
    """
    Extract html_context from function args/kwargs.
    
    Looks for:
    - kwargs["html_context"]
    - kwargs["context"]
    - Second positional arg (common pattern: page, html_context, ...)
    """
    # Check kwargs first
    if "html_context" in kwargs:
        return kwargs["html_context"]
    
    if "context" in kwargs:
        return kwargs["context"]
    
    # Check positional args (typically: page, html_context, coordinator, ...)
    if len(args) >= 2 and isinstance(args[1], dict):
        return args[1]
    
    # Not found or can't modify safely
    return None


def _save_failure_snapshot(args: tuple, kwargs: dict, exception: Exception):
    """
    Save HTML snapshot and context snapshot to failed_extractions/ directory.
    
    Saves:
    - HTML content (if page object available)
    - Context dict as JSON
    - Error details
    """
    try:
        from ..config import PROJECT_ROOT
        
        # Create failed_extractions directory
        snapshot_dir = Path(PROJECT_ROOT) / "uploads" / "failed_extractions"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Get session_id from args/kwargs
        session_id = kwargs.get("session_id")
        if not session_id and len(args) >= 4:
            session_id = args[3] if isinstance(args[3], str) else None
        
        # Fallback timestamp if no session_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = session_id if session_id else f"failed_{timestamp}"
        
        # Save HTML snapshot (if page object available)
        page = kwargs.get("page") or (args[0] if args else None)
        if page and hasattr(page, "content"):
            try:
                html_content = page.content()
                html_file = snapshot_dir / f"{base_name}.html"
                html_file.write_text(html_content, encoding="utf-8")
                logger.info(f"[snapshot] Saved HTML: {html_file}")
            except Exception as e:
                logger.warning(f"[snapshot] Could not save HTML: {e}")
        
        # Save context snapshot as JSON
        html_context = _get_html_context(args, kwargs)
        if html_context:
            try:
                import orjson
                context_file = snapshot_dir / f"{base_name}_context.json"
                context_file.write_bytes(orjson.dumps(html_context, option=orjson.OPT_INDENT_2))
                logger.info(f"[snapshot] Saved context: {context_file}")
            except Exception as e:
                logger.warning(f"[snapshot] Could not save context: {e}")
        
        # Save error details
        error_file = snapshot_dir / f"{base_name}_error.txt"
        error_details = f"""Extraction Failure Report
Generated: {datetime.now().isoformat()}
Session ID: {session_id or 'Unknown'}
Exception Type: {type(exception).__name__}
Exception Message: {str(exception)}

URL: {html_context.get('url') if html_context else 'Unknown'}
State: {html_context.get('state') if html_context else 'Unknown'}
County: {html_context.get('county') if html_context else 'Unknown'}

Full Traceback:
{_get_traceback_str()}
"""
        error_file.write_text(error_details, encoding="utf-8")
        logger.info(f"[snapshot] Saved error details: {error_file}")
    
    except Exception as e:
        logger.error(f"[snapshot] Failed to save failure snapshot: {e}")


def _log_extraction_failure(args: tuple, kwargs: dict, exception: Exception, attempts: int):
    """
    Log extraction failure to JSONL for pattern analysis and learning.
    """
    try:
        import orjson

        from ..config import LOG_DIR
        
        log_file = Path(LOG_DIR) / "extraction_failures.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        html_context = _get_html_context(args, kwargs)
        session_id = kwargs.get("session_id", "unknown")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "attempts": attempts,
            "url": html_context.get("url") if html_context else None,
            "state": html_context.get("state") if html_context else None,
            "county": html_context.get("county") if html_context else None,
            "handler": html_context.get("handler") if html_context else None,
        }
        
        # Append to log file
        with open(log_file, "ab") as f:
            f.write(orjson.dumps(log_entry))
            f.write(b"\n")
        
        logger.debug(f"[retry] Logged failure to {log_file}")
    
    except Exception as e:
        logger.warning(f"[retry] Could not log failure: {e}")


def _get_traceback_str() -> str:
    """Get current traceback as string."""
    try:
        import traceback
        return traceback.format_exc()
    except Exception:
        return "Traceback unavailable"


# Convenience decorator for common use case
retry_parse = retry_with_snapshot(max_attempts=3, backoff=2.0, snapshot_on_final_fail=True)


# Example usage function
def example_handler_with_retry():
    """
    Example showing how to use retry_with_snapshot in a handler.
    """
    @retry_with_snapshot(max_attempts=3, backoff=2.0)
    def parse_with_retry(page, html_context, coordinator, session_id, **kwargs):
        """Wrapped parse function with automatic retry."""
        # Your extraction logic here
        from ..handlers.formats.html_dynamic_fallback import parse as dynamic_parse
        return dynamic_parse(
            page=page,
            coordinator=coordinator,
            context=html_context,
            session_id=session_id,
            **kwargs,
        )
    
    # Usage:
    # result = parse_with_retry(page, html_context, coordinator, session_id)
    return None  # Placeholder
