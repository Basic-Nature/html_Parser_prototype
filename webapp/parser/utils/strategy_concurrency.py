"""
strategy_concurrency.py
Runs extraction strategies concurrently where safe.
DOM-sensitive strategies run sequentially; HTML/string strategies run in a thread pool.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

from .browser_utils import safe_content
from .logger_singleton import logger

StrategyResult = Tuple[List[str], List[Dict[str, Any]], Dict[str, Any]]
Strategy = Callable[[Any, dict | None], List[StrategyResult]]

def run_strategies_concurrently(
    page,
    context: dict,
    dom_strategies: List[Strategy],
    pure_html_strategies: List[Strategy],
    max_workers: int = 4
) -> List[StrategyResult]:
    results: List[StrategyResult] = []

    # Sequential DOM-first (to avoid Playwright race issues)
    for fn in dom_strategies:
        name = getattr(fn, "__name__", "dom_strategy")
        try:
            part = fn(page, context) or []
            for r in part:
                if r[0] and r[1]:
                    results.append(r)
        except Exception as e:
            logger.warning(f"[CONCURRENCY] DOM strategy {name} failed: {e}")

    # Snapshot HTML once
    html = ""
    try:
        html = safe_content(page) or ""
    except Exception:
        pass
    # Provide a lightweight shim page object for pure html strategies
    class HtmlShim:
        def __init__(self, raw): self._raw = raw
        def content(self): return self._raw
    shim = HtmlShim(html)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        fut_map = {}
        for fn in pure_html_strategies:
            name = getattr(fn, "__name__", "html_strategy")
            fut = pool.submit(_safe_run_strategy, fn, shim, context, name)
            fut_map[fut] = name
        for fut in as_completed(fut_map):
            name = fut_map[fut]
            try:
                part = fut.result() or []
                for r in part:
                    if r[0] and r[1]:
                        results.append(r)
            except Exception as e:
                logger.warning(f"[CONCURRENCY] Strategy {name} error: {e}")
    return results

def _safe_run_strategy(fn, page_like, context, name):
    try:
        return fn(page_like, context)
    except Exception as e:
        from .logger_singleton import logger
        logger.warning(f"[CONCURRENCY] {_safe_run_strategy.__name__} {name} failed: {e}")
        return []

async def run_strategies_concurrently_async(
    page,
    context: dict,
    dom_strategies: List[Strategy],
    pure_html_strategies: List[Strategy],
    max_workers: int = 4
) -> List[StrategyResult]:
    """
    Async variant.
    Current implementation: runs all strategies (sync) in default executor.
    DOM strategies executed sequentially (awaited) to avoid Playwright race issues.
    Pure HTML strategies dispatched concurrently via threads.
    If you later convert strategies to native async, detect with asyncio.iscoroutinefunction and await directly.
    """
    loop = asyncio.get_running_loop()
    results: List[StrategyResult] = []

    # Sequential DOM strategies (still run in executor to avoid blocking loop)
    for fn in dom_strategies:
        name = getattr(fn, "__name__", "dom_strategy")
        try:
            part = await loop.run_in_executor(None, partial(fn, page, context))
            for r in part or []:
                if r[0] and r[1]:
                    results.append(r)
        except Exception as e:
            logger.warning(f"[CONCURRENCY][ASYNC] DOM strategy {name} failed: {e}")

    # Snapshot HTML once (reuse existing safe_content if sync)
    try:
        html = safe_content(page) or ""
    except Exception:
        html = ""

    class HtmlShim:
        def __init__(self, raw): self._raw = raw
        def content(self): return self._raw
    shim = HtmlShim(html)

    async def _run(fn):
        name = getattr(fn, "__name__", "html_strategy")
        try:
            return await loop.run_in_executor(None, partial(fn, shim, context))
        except Exception as e:
            logger.warning(f"[CONCURRENCY][ASYNC] Strategy {name} error: {e}")
            return []

    tasks = [asyncio.create_task(_run(fn)) for fn in pure_html_strategies]
    for task in asyncio.as_completed(tasks):
        part = await task
        for r in part or []:
            if r[0] and r[1]:
                results.append(r)
    return results