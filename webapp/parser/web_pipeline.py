import os
import threading
import time
import traceback
from pathlib import Path

import orjson

from .config import (
    PIPELINE_HEARTBEAT_INTERVAL,
    PIPELINE_MAX_WORKERS,
    PROCESSED_URLS_FILE,
    SLOW_NLP_AUDIT_MIN_HITS,
    SLOW_NLP_AUDIT_THRESHOLD,
    URL_LIST_FILE,
)
from .html_election_parser import main
from .utils.logger_singleton import logger, prompt
from .utils.shared_logic import safe_clear, safe_is_set, safe_set


class CancellationManager(threading.Thread):
    """
    Manages cancellation flags per session/user.
    """
    def __init__(self) -> None:
        super().__init__()
        self._flags = {}
        self._lock = threading.Lock()
        self._unknown_warned = set()

    def get_flag(self, session_id) -> threading.Event:
        with self._lock:
            if session_id not in self._flags:
                self._flags[session_id] = threading.Event()
                # Only log to backend, not frontend
                print(f"[DEBUG] Created cancellation flag for session_id={session_id}")
            else:
                print(f"[DEBUG] Reusing cancellation flag for session_id={session_id}")
            return self._flags[session_id]

    def cancel(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                safe_set(self._flags[session_id])
                logger.info({
                    "level": "CANCELLED",
                    "type": "cancel",
                    "message": f"Cancellation requested (session_id={session_id})",
                    "session_id": session_id
                })
            else:
                logger.warning({
                    "level": "WARNING",
                    "type": "cancel",
                    "message": f"Cancellation requested for unknown session_id={session_id}",
                    "session_id": session_id
                })

    def reset(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                ev = self._flags[session_id]
                if ev.is_set():
                    safe_clear(ev)
                    print(f"[DEBUG] Cancellation flag reset for session_id={session_id}")
            else:
                if session_id not in self._unknown_warned:
                    self._unknown_warned.add(session_id)
                    logger.warning({
                        "level": "WARNING",
                        "type": "cancellation",
                        "message": f"Reset requested for unknown session_id={session_id}",
                        "session_id": session_id
                    })

    def remove(self, session_id) -> None:
        with self._lock:
            if session_id in self._flags:
                del self._flags[session_id]
                # Only log to backend, not frontend
                print(f"[DEBUG] Cancellation flag removed for session_id={session_id}")
            else:
                # Only emit to frontend if something is wrong
                if session_id not in self._unknown_warned:
                    self._unknown_warned.add(session_id)
                    logger.warning({
                        "level": "WARNING",
                        "type": "cancellation",
                        "message": f"Remove requested for unknown session_id={session_id}",
                        "session_id": session_id
                    })
             
# Instantiate globally
cancellation_manager = CancellationManager()

def heartbeat(session_id, cancel_flag, interval=10, emit_func=None):
    while True:
        time.sleep(interval)
        # Only emit heartbeat to frontend, don't log to terminal
        if emit_func:
            emit_func({
                "type": "heartbeat",
                "session_id": session_id,
                "status": "alive",
                "timestamp": time.time()
            })
        if safe_is_set(cancel_flag):
            break

def save_pipeline_report(session_id, results, errors):
    report_dir = os.path.join("output", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{session_id}.json")
    with open(report_path, "wb") as f:
        f.write(orjson.dumps({
            "results": results,
            "errors": errors
        }, option=orjson.OPT_INDENT_2))
    return report_path


def _collect_output_artifacts(entries) -> dict:
    output_root = Path("output").resolve()
    output_dirs: set[str] = set()
    csv_paths: set[str] = set()
    xlsx_paths: set[str] = set()
    metadata_paths: set[str] = set()
    other_paths: set[str] = set()

    def _to_rel_output_path(path_value):
        if not isinstance(path_value, str):
            return None
        raw = path_value.strip()
        if not raw:
            return None
        normalized = raw.replace("\\", "/")
        if normalized.startswith("output/"):
            normalized = normalized[len("output/"):]
        try:
            abs_path = os.path.abspath(raw)
            output_root_str = str(output_root)
            if abs_path == output_root_str:
                return None
            if abs_path.startswith(output_root_str + os.sep):
                return os.path.relpath(abs_path, output_root_str).replace("\\", "/")
        except Exception:
            pass
        if normalized.startswith("/"):
            return None
        return normalized

    def _record_path(path_value):
        rel = _to_rel_output_path(path_value)
        if not rel:
            return
        low = rel.lower()
        if low.endswith(".csv"):
            csv_paths.add(rel)
        elif low.endswith(".xlsx") or low.endswith(".xls"):
            xlsx_paths.add(rel)
        elif low.endswith("results.metadata.json") or low.endswith("metadata.json"):
            metadata_paths.add(rel)
        elif any(low.endswith(ext) for ext in (".json", ".pdf")):
            other_paths.add(rel)

    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            for key in ("output_file", "output_path", "csv_path", "xlsx_path", "excel_path", "metadata_path"):
                _record_path(entry.get(key))
                _record_path(metadata.get(key))
            rel_dir = _to_rel_output_path(entry.get("output_dir")) or _to_rel_output_path(metadata.get("output_dir"))
            if rel_dir:
                output_dirs.add(rel_dir.rstrip("/"))

    for rel_dir in sorted(output_dirs):
        try:
            abs_dir = output_root / rel_dir
            if not abs_dir.is_dir():
                continue
            for artifact in abs_dir.iterdir():
                if not artifact.is_file():
                    continue
                _record_path(str(artifact))
        except Exception:
            continue

    ordered_csv = sorted(csv_paths)
    ordered_xlsx = sorted(xlsx_paths)
    ordered_metadata = sorted(metadata_paths)
    ordered_other = sorted(other_paths)

    primary_download = None
    for bucket in (ordered_csv, ordered_xlsx, ordered_metadata, ordered_other):
        if bucket:
            primary_download = bucket[0]
            break

    return {
        "output_dirs": sorted(output_dirs),
        "csv": ordered_csv,
        "xlsx": ordered_xlsx,
        "metadata": ordered_metadata,
        "other": ordered_other,
        "primary_download": primary_download,
    }

def process_urls_for_web(
    prompt_queue,
    session_id,
    cancel_flag,
    max_workers=PIPELINE_MAX_WORKERS,
    emit_func=None,
    output_bypass=False,
    manual_source='input',
    disable_internal_heartbeat=False,
    force_parse_format=None,
    force_parse_input_file=None,
    principal=None,
    principal_source=None,
    dev_isolation_bypass=False,
    **kwargs
) -> None:
    """
    Single-run (interactive) pipeline:
      - Sets up logging / heartbeat
      - Invokes main() exactly once (interactive or with provided urls)
      - Passes prompt_queue and max_workers to main()
      - No per-URL threading, batching, summary aggregation, or prompt queue loop here.
      Batch / parallel logic is delegated to main() internally.
    """
    cancellation_manager.reset(session_id)

    logger.set_mode("webapp")
    logger.set_format("json")
    if emit_func:
        prompt.set_mode("webapp")
        prompt.set_socketio_emit_func(emit_func)
    
    # --- Multi-tenant isolation setup ---
    from .health.session_branching import get_isolated_branch, get_principal_tier

    if dev_isolation_bypass:
        logger.info({
            "level": "INFO",
            "type": "isolation",
            "message": "[MultiTenant] Dev isolation bypass enabled for localhost run.",
            "session_id": session_id,
            "principal": principal,
        })
        kwargs.setdefault("allowlist_bypass", True)
        kwargs.setdefault("trust_bypass", True)
        if emit_func:
            emit_func({
                "level": "INFO",
                "type": "isolation",
                "message": "Dev isolation bypass enabled for localhost run.",
                "session_id": session_id,
            })
    elif principal and principal_source:
        try:
            # Ensure principal has an isolation branch
            branch = get_isolated_branch(principal)
            if branch:
                tier = get_principal_tier(principal, principal_source)
                logger.info({
                    "level": "INFO",
                    "type": "isolation",
                    "message": f"[MultiTenant] Initialized isolation branch for principal (tier={tier.name})",
                    "session_id": session_id,
                    "principal": principal,
                    "principal_source": principal_source,
                    "privilege_tier": tier.value
                })
        except Exception as e:
            logger.warning({
                "level": "WARNING",
                "type": "isolation",
                "message": f"[MultiTenant] Failed to initialize isolation: {e}",
                "session_id": session_id,
                "principal": principal
            })

    if not disable_internal_heartbeat:
        threading.Thread(
            target=heartbeat,
            args=(session_id, cancel_flag, PIPELINE_HEARTBEAT_INTERVAL, emit_func),
            daemon=True
        ).start()

    # Emit structured run start to frontend if available
    if emit_func:
        try:
            emit_func({
                "type": "run_started",
                "session_id": session_id,
                "manual_source": manual_source,
                "output_bypass": bool(output_bypass),
                "timestamp": time.time(),
            })
        except Exception:
            pass

    logger.info({
        "level": "INFO",
        "type": "status",
        "message": f"Session started for {session_id}",
        "session_id": session_id
    })
    
    # Track if run_summary was emitted
    run_summary_emitted = False

    try:
        urls = kwargs.pop("urls", None)

        # Progress watcher: emit periodic run_progress events based on .processed_urls
        progress_stop = threading.Event()
        def _progress_watcher():
            from collections import Counter
            while not progress_stop.is_set():
                time.sleep(2)
                try:
                    entries = []
                    if PROCESSED_URLS_FILE.exists() and PROCESSED_URLS_FILE.stat().st_size > 0:
                        with open(PROCESSED_URLS_FILE, 'rb') as f:
                            entries = orjson.loads(f.read())
                    processed = len(entries) if isinstance(entries, list) else 0
                    statuses = Counter()
                    if isinstance(entries, list):
                        for e in entries:
                            if isinstance(e, dict):
                                statuses[e.get('status', 'unprocessed')] += 1
                    total_expected = 0
                    if isinstance(urls, list):
                        total_expected = len(urls)
                    else:
                        # try to infer from URL_LIST_FILE when interactive
                        try:
                            if os.path.exists(URL_LIST_FILE):
                                with open(URL_LIST_FILE, 'r', encoding='utf-8') as fh:
                                    total_expected = sum(1 for ln in fh if ln.strip() and not ln.strip().startswith('#'))
                        except Exception:
                            total_expected = 0
                    if emit_func:
                        try:
                            emit_func({
                                'type': 'run_progress',
                                'session_id': session_id,
                                'processed': processed,
                                'total_entries': total_expected,
                                'status_counts': dict(statuses),
                                'timestamp': time.time(),
                            })
                        except Exception:
                            pass
                except Exception:
                    pass

        if emit_func:
            watcher_thread = threading.Thread(target=_progress_watcher, daemon=True)
            watcher_thread.start()

        # Always pass prompt_queue and max_workers to main()
        main_kwargs = dict(
            session_id=session_id,
            cancel_flag=cancel_flag,
            output_bypass=output_bypass,
            manual_source=manual_source,
            force_parse_input_file=force_parse_input_file,
            force_parse_format=force_parse_format,
            principal=principal,
            principal_source=principal_source,
            # If uploads is selected, do not fall back to URL list on failure
            continue_on_override_failure=False if manual_source == 'uploads' else True,
            prompt_queue=prompt_queue,
            max_workers=max_workers,
            **kwargs
        )

        if urls is None:
            # Interactive / internal URL selection path (main() handles listing & prompts)
            try:
                if os.path.exists(URL_LIST_FILE):
                    with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
                        raw_urls = [
                            ln.strip() for ln in f
                            if ln.strip() and not ln.strip().startswith("#")
                        ]
                else:
                    raw_urls = []
                if not raw_urls:
                    logger.error({
                        "level": "ERROR",
                        "type": "input",
                        "message": "urls.txt has no usable URLs (aborting before interactive main()).",
                        "session_id": session_id
                    })
                    logger.info({
                        "level": "INFO",
                        "type": "input",
                        "message": f"Edit file at: {URL_LIST_FILE}",
                        "session_id": session_id
                    })
                    cancellation_manager.remove(session_id)
                    return
            except Exception as e:
                logger.error({
                    "level": "ERROR",
                    "type": "exception",
                    "message": f"Failed preparing URL list: {e}",
                    "session_id": session_id,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                cancellation_manager.remove(session_id)
                return

            # --- Pre-processing: Validate raw_urls against principal's isolation ---
            if dev_isolation_bypass and raw_urls:
                logger.info({
                    "level": "INFO",
                    "type": "isolation",
                    "message": f"[MultiTenant] Dev isolation bypass active; skipping isolation filter for {len(raw_urls)} URL(s).",
                    "session_id": session_id,
                    "principal": principal,
                    "url_count": len(raw_urls)
                })
            elif principal and principal_source and raw_urls:
                try:
                    from .health.session_branching import validate_url_access
                    blocked_urls = []
                    for url in raw_urls:
                        allowed, reason = validate_url_access(principal, url, "view", principal_source)
                        if not allowed:
                            blocked_urls.append((url, reason))
                            logger.warning({
                                "level": "WARNING",
                                "type": "isolation",
                                "message": f"[MultiTenant] URL blocked due to isolation: {reason}",
                                "session_id": session_id,
                                "principal": principal,
                                "url": url,
                                "block_reason": reason
                            })
                    
                    if blocked_urls:
                        logger.warning({
                            "level": "WARNING",
                            "type": "isolation",
                            "message": f"[MultiTenant] {len(blocked_urls)} URL(s) filtered by isolation policy",
                            "session_id": session_id,
                            "principal": principal,
                            "blocked_count": len(blocked_urls)
                        })
                        # Filter out blocked URLs
                        raw_urls = [url for url in raw_urls if url not in [b[0] for b in blocked_urls]]
                        
                        if not raw_urls:
                            logger.error({
                                "level": "ERROR",
                                "type": "isolation",
                                "message": "[MultiTenant] No URLs remain after isolation filtering",
                                "session_id": session_id,
                                "principal": principal
                            })
                            cancellation_manager.remove(session_id)
                            return
                except Exception as e:
                    logger.error({
                        "level": "ERROR",
                        "type": "isolation",
                        "message": f"[MultiTenant] Isolation validation failed: {e}",
                        "session_id": session_id,
                        "principal": principal
                    })

            main(**main_kwargs)
        else:
            # Explicit URLs provided (pass through to main; let it batch internally)
            if isinstance(urls, str):
                urls = [urls]
            if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
                logger.error({
                    "level": "ERROR",
                    "type": "input",
                    "message": "Invalid 'urls' argument (must be list[str] or str).",
                    "session_id": session_id
                })
                cancellation_manager.remove(session_id)
                return

            # --- Pre-processing: Validate explicit URLs against principal's isolation ---
            if dev_isolation_bypass and urls:
                logger.info({
                    "level": "INFO",
                    "type": "isolation",
                    "message": f"[MultiTenant] Dev isolation bypass active; skipping isolation filter for {len(urls)} URL(s).",
                    "session_id": session_id,
                    "principal": principal,
                    "url_count": len(urls)
                })
            elif principal and principal_source and urls:
                try:
                    from .health.session_branching import validate_url_access
                    blocked_urls = []
                    for url in urls:
                        allowed, reason = validate_url_access(principal, url, "view", principal_source)
                        if not allowed:
                            blocked_urls.append((url, reason))
                            logger.warning({
                                "level": "WARNING",
                                "type": "isolation",
                                "message": f"[MultiTenant] URL blocked due to isolation: {reason}",
                                "session_id": session_id,
                                "principal": principal,
                                "url": url,
                                "block_reason": reason
                            })
                    
                    if blocked_urls:
                        logger.warning({
                            "level": "WARNING",
                            "type": "isolation",
                            "message": f"[MultiTenant] {len(blocked_urls)} URL(s) filtered by isolation policy",
                            "session_id": session_id,
                            "principal": principal,
                            "blocked_count": len(blocked_urls)
                        })
                        # Filter out blocked URLs
                        urls = [url for url in urls if url not in [b[0] for b in blocked_urls]]
                        
                        if not urls:
                            logger.error({
                                "level": "ERROR",
                                "type": "isolation",
                                "message": "[MultiTenant] No URLs remain after isolation filtering",
                                "session_id": session_id,
                                "principal": principal
                            })
                            cancellation_manager.remove(session_id)
                            return
                except Exception as e:
                    logger.error({
                        "level": "ERROR",
                        "type": "isolation",
                        "message": f"[MultiTenant] Isolation validation failed: {e}",
                        "session_id": session_id,
                        "principal": principal
                    })

            logger.info({
                "level": "INFO",
                "type": "status",
                "message": f"Dispatching main() with {len(urls)} provided URL(s).",
                "count": len(urls),
                "session_id": session_id
            })

            main(
                urls=urls,
                emit_func=emit_func,
                skip_url_prompt=True,
                url_source_label="direct override",
                **main_kwargs
            )
        # stop watcher (if running) after main returns
        try:
            progress_stop.set()
        except Exception:
            pass
        # Completion (single-run)
        if safe_is_set(cancel_flag):
            logger.info({
                "level": "CANCELLED",
                "type": "cancel",
                "message": "Run cancelled.",
                "session_id": session_id
            })
        else:
            logger.info({
                "level": "SUMMARY",
                "type": "summary",
                "message": "Single-run main() completed.",
                "session_id": session_id
            })
            # Try to summarize processed URLs and emit a concise report
            try:
                results = {}
                errors_list = []
                flagged_count = 0
                flagged_details = []
                confidences = []
                if PROCESSED_URLS_FILE.exists() and PROCESSED_URLS_FILE.stat().st_size > 0:
                    with open(PROCESSED_URLS_FILE, 'rb') as f:
                        entries = orjson.loads(f.read())
                    # entries expected to be a list of dicts with 'status'
                    from collections import Counter
                    statuses = Counter()
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        status = e.get('status', 'unprocessed')
                        statuses[status] += 1
                        # collect errors
                        if status in ('fail', 'error'):
                            err = {
                                'url': e.get('url'),
                                'status': status,
                                'timestamp': e.get('timestamp')
                            }
                            if 'error' in e:
                                err['error'] = e.get('error')
                            errors_list.append(err)
                        # flagged entries: count and collect detailed info when available
                        is_flagged = False
                        if e.get('flagged_for_review') or e.get('flagged') or e.get('flagged_suspicious'):
                            is_flagged = True
                            flagged_count += 1
                        if is_flagged:
                            detail = {
                                'url': e.get('url'),
                                'timestamp': e.get('timestamp'),
                                'status': status,
                            }
                            # gather human-friendly reasons if present
                            reasons = []
                            if isinstance(e.get('flagged_reason'), str):
                                reasons.append(e.get('flagged_reason'))
                            if isinstance(e.get('flagged_reasons'), (list, tuple)):
                                reasons.extend(e.get('flagged_reasons'))
                            if isinstance(e.get('flagged_suspicious'), (list, tuple)):
                                reasons.extend(e.get('flagged_suspicious'))
                            # AI analysis may produce 'flagged' list inside metadata
                            if isinstance(e.get('metadata'), dict):
                                md = e.get('metadata', {})
                                if isinstance(md.get('flagged_suspicious'), (list, tuple)):
                                    reasons.extend(md.get('flagged_suspicious'))
                                if isinstance(md.get('flagged_reason'), str):
                                    reasons.append(md.get('flagged_reason'))
                            if reasons:
                                # dedupe and keep short
                                seenr = []
                                for r in reasons:
                                    if r and r not in seenr:
                                        seenr.append(r)
                                detail['reasons'] = seenr[:5]
                            # include a small metadata excerpt for context
                            meta_excerpt = {}
                            md = e.get('metadata') if isinstance(e.get('metadata'), dict) else {}
                            for k in ('handler', 'contest', 'state', 'county', 'output_file', 'handler_args'):
                                if k in md:
                                    meta_excerpt[k] = md[k]
                            # quality metrics if present
                            if isinstance(md.get('quality_metrics'), dict) and md.get('quality_metrics').get('extraction_confidence') is not None:
                                meta_excerpt['extraction_confidence'] = md.get('quality_metrics').get('extraction_confidence')
                            if meta_excerpt:
                                detail['metadata_excerpt'] = meta_excerpt
                            flagged_details.append(detail)
                        # try to extract confidence from multiple potential locations
                        conf = None
                        if isinstance(e.get('quality_metrics'), dict):
                            conf = e.get('quality_metrics', {}).get('extraction_confidence')
                        if conf is None and isinstance(e.get('metadata'), dict):
                            conf = e.get('metadata', {}).get('quality_metrics', {}).get('extraction_confidence')
                        # some handlers may store top-level 'confidence' keys
                        if conf is None:
                            conf = e.get('extraction_confidence') or e.get('confidence')
                        try:
                            if conf is not None:
                                confidences.append(float(conf))
                        except Exception:
                            pass

                    results = {
                        "total_entries": len(entries),
                        "status_counts": dict(statuses),
                        "sample_recent": entries[-10:] if isinstance(entries, list) else [],
                        "errors": errors_list,
                        "flagged_count": flagged_count,
                        "flagged_details": flagged_details,
                    }
                else:
                    results = {"total_entries": 0, "status_counts": {}, "sample_recent": [], "errors": [], "flagged_count": 0}

                artifacts = _collect_output_artifacts(entries if isinstance(entries, list) else [])

                audit_hits = []
                if "entries" in locals() and isinstance(entries, list):
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        md = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
                        audit_signals = md.get("audit_signals") or e.get("audit_signals")
                        if not isinstance(audit_signals, dict):
                            continue
                        score = audit_signals.get("audit_weighted_score")
                        try:
                            score_val = float(score) if score is not None else None
                        except Exception:
                            score_val = None
                        if score_val is not None and score_val >= SLOW_NLP_AUDIT_THRESHOLD:
                            audit_hits.append({
                                "url": e.get("url"),
                                "audit_signals": audit_signals,
                                "metadata": md,
                            })

                if audit_hits and len(audit_hits) >= SLOW_NLP_AUDIT_MIN_HITS:
                    def _run_slow_nlp_audit():
                        try:
                            from .health.health_router import get_learning_engine
                            from .utils.ml_telemetry import record_ml_event
                            engine = get_learning_engine()
                            for hit in audit_hits:
                                md = hit.get("metadata") or {}
                                session_context = {
                                    "state": md.get("state"),
                                    "county": md.get("county"),
                                    "contest": md.get("contest"),
                                    "handler": md.get("handler"),
                                    "url": hit.get("url"),
                                }
                                engine.ingest_training_signal(
                                    session_context,
                                    success=False,
                                    quality_metrics=hit.get("audit_signals") or {},
                                )
                                record_ml_event(
                                    "learning_engine",
                                    "ingest_training_signal",
                                    session_id=session_id,
                                    metadata={
                                        "url": hit.get("url"),
                                        "handler": md.get("handler"),
                                        "state": md.get("state"),
                                        "county": md.get("county"),
                                    },
                                )
                            logger.warning({
                                "level": "WARNING",
                                "type": "audit",
                                "message": "[SlowNLPAudit] Session-level audit completed.",
                                "session_id": session_id,
                                "audit_hit_count": len(audit_hits),
                                "threshold": SLOW_NLP_AUDIT_THRESHOLD,
                            })
                            record_ml_event(
                                "learning_engine",
                                "slow_nlp_audit_completed",
                                session_id=session_id,
                                metadata={
                                    "audit_hit_count": len(audit_hits),
                                    "threshold": SLOW_NLP_AUDIT_THRESHOLD,
                                },
                            )
                            if emit_func:
                                emit_func({
                                    "type": "slow_nlp_audit",
                                    "session_id": session_id,
                                    "status": "completed",
                                    "audit_hit_count": len(audit_hits),
                                    "threshold": SLOW_NLP_AUDIT_THRESHOLD,
                                    "timestamp": time.time(),
                                })
                        except Exception as exc:
                            logger.warning({
                                "level": "WARNING",
                                "type": "audit",
                                "message": f"[SlowNLPAudit] Session audit failed: {exc}",
                                "session_id": session_id,
                            })

                    try:
                        audit_thread = threading.Thread(target=_run_slow_nlp_audit, daemon=True)
                        audit_thread.start()
                        if emit_func:
                            emit_func({
                                "type": "slow_nlp_audit",
                                "session_id": session_id,
                                "status": "started",
                                "audit_hit_count": len(audit_hits),
                                "threshold": SLOW_NLP_AUDIT_THRESHOLD,
                                "timestamp": time.time(),
                            })
                    except Exception:
                        pass

                # compute confidence metrics
                conf_metrics = {}
                if confidences:
                    try:
                        import statistics
                        conf_metrics = {
                            'count': len(confidences),
                            'avg': float(sum(confidences) / len(confidences)),
                            'min': float(min(confidences)),
                            'max': float(max(confidences)),
                            'median': float(statistics.median(confidences)),
                        }
                    except Exception:
                        conf_metrics = {'count': len(confidences)}
                else:
                    conf_metrics = {'count': 0}

                results['confidence_metrics'] = conf_metrics
                results['artifacts'] = artifacts

                report_path = save_pipeline_report(session_id, results, errors=[])
                run_summary_emitted = True
                if isinstance(report_path, str) and report_path:
                    rel_report = report_path.replace("\\", "/")
                    if rel_report.startswith("output/"):
                        rel_report = rel_report[len("output/"):]
                    if rel_report and rel_report not in artifacts['other']:
                        artifacts['other'].append(rel_report)
                if emit_func:
                    try:
                        emit_func({
                            "type": "run_summary",
                            "session_id": session_id,
                            "summary": results,
                            "artifacts": artifacts,
                            "report_path": report_path,
                            "timestamp": time.time(),
                        })
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning({
                    "level": "WARNING",
                    "type": "summary",
                    "message": f"Failed to build run summary: {exc}",
                    "session_id": session_id
                })

    except Exception as e:
        logger.error({
            "level": "ERROR",
            "type": "exception",
            "message": f"Unhandled exception in process_urls_for_web: {e}",
            "session_id": session_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    finally:
        logger.info({
            "level": "INFO",
            "type": "cleanup",
            "message": f"[Cleanup] Finally block executing. run_summary_emitted={run_summary_emitted}, emit_func={'present' if emit_func else 'None'}",
            "session_id": session_id
        })
        # Emit minimal run_summary if not already emitted
        if emit_func and not run_summary_emitted:
            try:
                logger.info({
                    "level": "INFO",
                    "type": "fallback",
                    "message": "[FallbackReport] Generating minimal pipeline report (run_summary not emitted during processing)",
                    "session_id": session_id
                })
                # Try to load processed_urls for minimal status
                from .Context_Integration.context_organizer import ContextOrganizer
                organizer = ContextOrganizer()
                processed_urls = organizer.get_processed_urls() or []
                status_counts = {}
                for entry in processed_urls[-20:]:  # Last 20 for efficiency
                    status = entry.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                minimal_results = {
                    "total_entries": len(processed_urls),
                    "status_counts": status_counts,
                    "sample_recent": processed_urls[-5:] if processed_urls else [],
                    "errors": [],
                    "flagged_count": 0,
                    "confidence_metrics": {"count": 0},
                }
                fallback_artifacts = _collect_output_artifacts(processed_urls if isinstance(processed_urls, list) else [])
                minimal_results["artifacts"] = fallback_artifacts
                
                report_path = save_pipeline_report(session_id, minimal_results, errors=[])
                logger.info({
                    "level": "INFO",
                    "type": "fallback",
                    "message": f"[FallbackReport] Report saved: {report_path}",
                    "session_id": session_id
                })
                emit_func({
                    "type": "run_summary",
                    "session_id": session_id,
                    "summary": minimal_results,
                    "artifacts": fallback_artifacts,
                    "report_path": report_path,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning({
                    "level": "WARNING",
                    "type": "fallback",
                    "message": f"[FallbackReport] Failed to emit fallback report: {e}",
                    "session_id": session_id
                })
        
        # --- Clean up multi-tenant isolation on session end ---
        if principal and principal_source:
            try:
                from .health.session_branching import cleanup_principal_isolation
                cleanup_principal_isolation(principal)
                logger.info({
                    "level": "INFO",
                    "type": "isolation",
                    "message": "[MultiTenant] Principal isolation branch cleaned up on session end",
                    "session_id": session_id,
                    "principal": principal
                })
            except Exception as e:
                logger.warning({
                    "level": "WARNING",
                    "type": "isolation",
                    "message": f"[MultiTenant] Cleanup failed: {e}",
                    "session_id": session_id,
                    "principal": principal
                })
        
        cancellation_manager.remove(session_id)

def cancel_processing(session_id) -> None:
    cancellation_manager.cancel(session_id)
    logger.info({
        "level": "CANCELLED",
        "type": "cancel",
        "message": f"Cancellation requested for session_id={session_id}",
        "session_id": session_id
    })

