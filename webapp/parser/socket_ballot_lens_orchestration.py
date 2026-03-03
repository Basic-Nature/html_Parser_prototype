from __future__ import annotations

from typing import Any


def _normalize_payload(data) -> dict[str, Any]:
    return data if isinstance(data, dict) else {}


def _initialize_session_and_auth(payload: dict[str, Any], h: dict[str, Any]) -> dict[str, Any] | None:
    h["cleanup_sessions"]()
    dev_isolation_bypass = h["is_dev_isolation_bypass_request"]()
    session_id = h["resolve_session_id"](payload, create_if_missing=True)
    if not isinstance(session_id, str):
        h["logger"].error({
            "level": "ERROR",
            "type": "status",
            "message": "Unable to resolve session_id.",
            "session_id": None,
        })
        return None

    if not h["rate_limit_socket_action"](session_id, "ballot_lens"):
        h["emit"](
            "parser_output",
            h["normalize_log_obj"]({
                "level": "WARNING",
                "type": "status",
                "message": "Rate limit exceeded for starting a job.",
                "session_id": session_id,
            }),
            room=session_id,
        )
        return None

    principal, principal_source, cert_metadata = h["get_request_principal"]()
    arr_cert_header = h["request"].headers.get("X-ARR-ClientCert", "") or ""
    principal_kind = None
    if isinstance(principal, str):
        principal_kind = "cert" if principal.startswith("cert:") else "principal"

    gate_ok = h["require_cert_for_socket_action"]("ballot_lens", session_id=session_id)
    h["logger"].info({
        "level": "INFO",
        "type": "auth",
        "message": "Socket cert gate decision for ballot_lens.",
        "session_id": session_id,
        "cert_gate": "allow" if gate_ok else "deny",
        "require_cert": h["require_cert_for_mutations"],
        "arr_client_cert_present": bool(arr_cert_header),
        "arr_client_cert_len": len(arr_cert_header) if arr_cert_header else 0,
        "principal_present": bool(principal),
        "principal_kind": principal_kind,
        "principal_source": principal_source,
        "cert_metadata_error": bool(cert_metadata)
        and isinstance(cert_metadata, dict)
        and bool(cert_metadata.get("error")),
    })
    if not gate_ok:
        return None

    h["join_room"](session_id)
    h["socketio"].sleep(0.25)
    h["emit"]("session_id", {"session_id": session_id})

    try:
        socket_sid = h["safe_sid"]()
    except Exception:
        socket_sid = getattr(h["request"], "sid", None)
    if isinstance(socket_sid, str):
        h["session_manager"].bind_socket(socket_sid, session_id)

    if not h["session_manager"].has_session(session_id):
        h["create_session_metadata"](session_id)
    meta = h["session_manager"].get_metadata(session_id) or {}
    h["session_manager"].mark_active(session_id)
    h["session_manager"].touch_session(session_id)
    h["session_manager"].update_metadata(session_id, dev_isolation_bypass=dev_isolation_bypass)

    if meta.get("auth_blocked"):
        h["emit"](
            "parser_output",
            h["normalize_log_obj"]({
                "level": "INFO",
                "type": "auth",
                "message": "Session blocked due to certificate change. Re-authenticate to continue.",
                "session_id": session_id,
            }),
            room=session_id,
        )
        return None

    if h["safe_get"](meta, "locked") and h["safe_is_alive"](session_id):
        h["logger"].error({
            "level": "ERROR",
            "type": "status",
            "message": "Session is locked. Wait for current job to finish.",
            "session_id": session_id,
        })
        return None
    if h["safe_is_alive"](session_id):
        h["logger"].warning({
            "level": "WARNING",
            "type": "status",
            "message": "Parser already running for this session.",
            "session_id": session_id,
        })
        return None

    return {
        "session_id": session_id,
        "dev_isolation_bypass": dev_isolation_bypass,
        "principal": principal,
        "principal_source": principal_source,
    }


def _prepare_run_inputs(payload: dict[str, Any], session_id: str, dev_isolation_bypass: bool, h: dict[str, Any]) -> dict[str, Any]:
    requested_source = h["safe_lower"](
        h["safe_get"](payload, "file_source", h["get_manual_source"](session_id))
    )
    if requested_source not in {"input", "uploads"}:
        requested_source = "input"

    requested_origin = h["safe_lower"](h["safe_get"](payload, "manual_source_origin", None))
    if requested_origin not in {"user", "default", "server"}:
        requested_origin = (
            "user"
            if h["safe_get"](payload, "file_source") == "uploads"
            else h["session_manager"].get_manual_source_origin(session_id)
        )

    force_parse_input_file = None
    force_parse_format = None
    manual_upload_rel = None
    manual_upload_name = h["safe_strip"](h["safe_get"](payload, "manual_upload_name", ""))
    raw_manual_upload_path = h["safe_strip"](h["safe_get"](payload, "manual_upload_path", ""))
    warehouse_override_url = h["safe_strip"](h["safe_get"](payload, "warehouse_override_url", ""))

    abs_uploads_dir = h["os"].path.abspath(h["uploads_dir"])
    if raw_manual_upload_path:
        normalized_rel = raw_manual_upload_path.replace("\\", "/").strip("/")
        candidate_path = h["os"].path.normpath(h["os"].path.join(abs_uploads_dir, normalized_rel))
        if candidate_path.startswith(abs_uploads_dir) and h["os"].path.isfile(candidate_path):
            manual_upload_rel = normalized_rel
            if not manual_upload_name:
                manual_upload_name = h["os"].path.basename(candidate_path)
            requested_source = "uploads"
            requested_origin = "user"
            force_parse_input_file = manual_upload_rel
            guessed_ext = ""
            try:
                _, ext = h["os"].path.splitext(manual_upload_name or manual_upload_rel)
                guessed_ext = h["safe_lower"](ext.lstrip("."))
            except Exception:
                guessed_ext = ""
            if guessed_ext:
                force_parse_format = guessed_ext
            h["session"]["FORCE_PARSE_INPUT_FILE"] = manual_upload_rel
            h["session"]["FORCE_PARSE_FORMAT"] = force_parse_format or guessed_ext or ""
            h["session"]["manual_source_pref"] = "uploads"
            h["logger"].info({
                "level": "INFO",
                "type": "manual_override",
                "message": f"[ManualOverride] Using uploaded file: {manual_upload_rel}",
                "session_id": session_id,
            })
        else:
            h["logger"].warning({
                "level": "WARNING",
                "type": "manual_override",
                "message": f"[ManualOverride] Invalid manual upload selection: {raw_manual_upload_path}",
                "session_id": session_id,
            })

    if requested_source == "uploads" and force_parse_input_file is None:
        force_parse_input_file = h["session"].get("FORCE_PARSE_INPUT_FILE")
        force_parse_format = h["session"].get("FORCE_PARSE_FORMAT")

    raw_direct_urls = h["safe_get"](payload, "direct_urls", [])
    direct_urls: list[str] = []
    if isinstance(raw_direct_urls, list):
        for entry in raw_direct_urls:
            url_text = h["safe_strip"](entry)
            if not url_text:
                continue
            try:
                parsed = h["urlparse"](url_text)
            except Exception:
                parsed = None
            if (
                not parsed
                or parsed.scheme not in {"http", "https"}
                or parsed.username
                or parsed.password
            ):
                h["logger"].warning({
                    "level": "WARNING",
                    "type": "input",
                    "message": f"Ignoring invalid direct URL: {url_text}",
                    "session_id": session_id,
                })
                continue
            allowed, reason = h["safe_validate_external_url"](
                url_text,
                allowlist_suffixes=h["url_allowlist_suffixes"],
                allowlist_hosts=h["url_allowlist_hosts"],
                enforce_allowlist=h["url_enforce_allowlist"],
                block_private_ips=h["url_block_private_ips"],
                allowlist_bypass=dev_isolation_bypass,
            )
            if not allowed:
                h["logger"].warning({
                    "level": "WARNING",
                    "type": "input",
                    "message": f"Blocked direct URL: {reason}",
                    "session_id": session_id,
                    "url": url_text,
                })
                continue
            direct_urls.append(url_text)

    if len(direct_urls) > h["direct_url_limit"]:
        h["logger"].warning({
            "level": "WARNING",
            "type": "input",
            "message": f"Direct URL list trimmed to {h['direct_url_limit']} entries.",
            "session_id": session_id,
        })
        direct_urls = direct_urls[: h["direct_url_limit"]]

    if direct_urls and requested_source == "uploads":
        h["logger"].warning({
            "level": "WARNING",
            "type": "input",
            "message": "Direct URLs ignored because manual uploads source is active.",
            "session_id": session_id,
        })
        direct_urls = []

    if not direct_urls and warehouse_override_url:
        allowed, reason = h["safe_validate_external_url"](
            warehouse_override_url,
            allowlist_suffixes=h["url_allowlist_suffixes"],
            allowlist_hosts=h["url_allowlist_hosts"],
            enforce_allowlist=h["url_enforce_allowlist"],
            block_private_ips=h["url_block_private_ips"],
            allowlist_bypass=dev_isolation_bypass,
        )
        if allowed:
            guard_ok, guard_reason = h["guarded_ingestion_allowed"]("direct_urls")
            if guard_ok:
                direct_urls = [warehouse_override_url]
                h["logger"].info({
                    "level": "INFO",
                    "type": "input",
                    "message": "Using warehouse override URL as direct URL fallback.",
                    "session_id": session_id,
                    "warehouse_override_url": warehouse_override_url,
                })
            else:
                h["logger"].warning({
                    "level": "WARNING",
                    "type": "security",
                    "message": f"Warehouse override fallback blocked by guarded gate: {guard_reason}",
                    "session_id": session_id,
                })
        else:
            h["logger"].warning({
                "level": "WARNING",
                "type": "input",
                "message": f"Warehouse override URL failed validation: {reason}",
                "session_id": session_id,
                "warehouse_override_url": warehouse_override_url,
            })

    if direct_urls:
        guard_ok, guard_reason = h["guarded_ingestion_allowed"]("direct_urls")
        if not guard_ok:
            h["logger"].warning({
                "level": "WARNING",
                "type": "security",
                "message": f"Direct URL ingestion blocked by guarded gate: {guard_reason}",
                "session_id": session_id,
            })
            direct_urls = []
        h["logger"].info({
            "level": "INFO",
            "type": "input",
            "message": f"Direct URL override engaged with {len(direct_urls)} link(s).",
            "session_id": session_id,
            "urls": direct_urls,
        })

    url_reference_hints = []
    if direct_urls:
        try:
            url_reference_hints = [h["collect_url_reference_hint"](url) for url in direct_urls]
            production_hits = sum(
                1
                for item in url_reference_hints
                if h["safe_get"](item.get("production", {}), "exists", False)
            )
            warehouse_hits = sum(
                1
                for item in url_reference_hints
                if int(h["safe_get"](item.get("warehouse", {}), "row_count", 0) or 0) > 0
            )
            output_hits = sum(
                1 for item in url_reference_hints if isinstance(item.get("output_match"), dict)
            )
            h["session_manager"].update_metadata(
                session_id,
                direct_url_reference_count=len(url_reference_hints),
                direct_url_reference_hits={
                    "production": production_hits,
                    "warehouse": warehouse_hits,
                    "output": output_hits,
                },
                direct_url_reference_preview=url_reference_hints[:5],
            )
            h["emit"](
                "parser_output",
                h["normalize_log_obj"]({
                    "level": "INFO",
                    "type": "input",
                    "message": (
                        f"Reference gate prepared for {len(url_reference_hints)} URL(s): "
                        f"production={production_hits}, warehouse={warehouse_hits}, output={output_hits}."
                    ),
                    "session_id": session_id,
                    "reference_hits": {
                        "production": production_hits,
                        "warehouse": warehouse_hits,
                        "output": output_hits,
                    },
                }),
                room=session_id,
            )
        except Exception as exc:
            h["logger"].warning({
                "level": "WARNING",
                "type": "input",
                "message": f"Reference gate collection failed: {exc}",
                "session_id": session_id,
            })

    if warehouse_override_url:
        h["logger"].info({
            "level": "INFO",
            "type": "status",
            "message": "Warehouse override acknowledged for parse run.",
            "session_id": session_id,
            "warehouse_override_url": warehouse_override_url,
        })

    return {
        "requested_source": requested_source,
        "requested_origin": requested_origin,
        "force_parse_input_file": force_parse_input_file,
        "force_parse_format": force_parse_format,
        "manual_upload_rel": manual_upload_rel,
        "warehouse_override_url": warehouse_override_url,
        "direct_urls": direct_urls,
        "url_reference_hints": url_reference_hints,
    }


def _configure_logging_and_prompt(session_id: str, h: dict[str, Any]) -> None:
    h["session_manager"].register_emitter(session_id, h["socketio_emit_func"])
    h["logger"].set_mode("webapp")
    h["logger"].set_format("json")

    def filtered_emit(line):
        try:
            obj = h["orjson"].loads(line) if isinstance(line, str) and line.strip().startswith("{") else None
        except Exception:
            obj = None
        lvl = (obj or {}).get("level") or ""
        if lvl.upper() in h["webapp_console_levels"]:
            h["logger"].enable_console_echo_webapp(True)
        else:
            h["logger"].enable_console_echo_webapp(False)
        h["socketio_emit_func"](line)

    h["logger"].set_socketio_emit_func(filtered_emit)
    h["prompt"].set_mode("webapp")
    h["prompt"].set_socketio_emit_func(
        lambda msg: h["socketio"].emit(
            "parser_output",
            h["normalize_log_obj"](
                msg
                if isinstance(msg, dict)
                else {
                    "level": "info",
                    "type": "prompt",
                    "message": str(msg),
                    "session_id": session_id,
                }
            ),
            room=session_id,
        )
    )


def _snapshot_output_artifacts(output_dir: Any) -> dict[str, float]:
    artifacts: dict[str, float] = {}
    try:
        if not output_dir.exists():
            return artifacts
        exts = {".csv", ".xlsx", ".xls", ".json", ".pdf"}
        for path in output_dir.rglob("*"):
            try:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in exts:
                    continue
                rel = str(path.relative_to(output_dir)).replace("\\", "/")
                artifacts[rel] = path.stat().st_mtime
            except Exception:
                continue
    except Exception:
        return artifacts
    return artifacts


def _detect_new_artifacts(
    artifacts_before: dict[str, float],
    artifacts_after: dict[str, float],
    start_time: float,
) -> list[str]:
    new_artifacts = [rel for rel in artifacts_after.keys() if rel not in artifacts_before]
    if new_artifacts:
        return new_artifacts
    return [
        rel
        for rel, mtime in artifacts_after.items()
        if mtime >= (start_time - 1.0)
    ]


def _emit_download_ready_for_rel(
    session_id: str,
    output_dir: Any,
    rel_path: str,
    h: dict[str, Any],
) -> bool:
    abs_path = output_dir / rel_path
    try:
        size = abs_path.stat().st_size
    except Exception:
        size = None
    try:
        h["emit_download_ready"](
            session_id,
            {
                "session_id": session_id,
                "filename": h["os"].path.basename(rel_path),
                "output_path": rel_path,
                "root": "output",
                "size": size,
                "source": "pipeline_output",
            },
        )
        return True
    except Exception:
        return False


def _finalize_worker_session(
    session_id: str,
    status: str,
    err: str | None,
    run_id: str,
    requested_source: str,
    requested_origin: str,
    output_bypass_flag: bool,
    manual_upload_rel: str | None,
    direct_urls: list[str],
    warehouse_override_url: str,
    start_time: float,
    h: dict[str, Any],
) -> None:
    duration_ms = int((h["time"].time() - start_time) * 1000)
    h["log_run_event"](
        {
            "type": "end",
            "run_id": run_id,
            "session_id": session_id,
            "ts": h["datetime"].now(h["timezone"].utc).isoformat(),
            "source": requested_source,
            "output_bypass": output_bypass_flag,
            "status": status,
            "error": err,
            "duration_ms": duration_ms,
            "manual_upload": manual_upload_rel,
            "direct_url_count": len(direct_urls),
        }
    )
    h["session_manager"].pop_thread(session_id)
    final_state = h["session_state"].COMPLETED
    cancel_flag = h["cancellation_manager"].get_flag(session_id)
    if h["safe_is_set"](cancel_flag):
        final_state = h["session_state"].CANCELLED
    elif status != "ok":
        final_state = h["session_state"].ERROR
    extras = {
        "manual_source": requested_source,
        "manual_source_origin": requested_origin,
        "run_id": run_id,
        "output_bypass": output_bypass_flag,
        "manual_upload_file": manual_upload_rel,
        "direct_url_count": len(direct_urls),
        "direct_urls": direct_urls,
        "warehouse_override_url": warehouse_override_url or None,
    }
    if err:
        extras["last_error"] = err
    h["transition_session"](
        session_id,
        final_state,
        locked=False,
        phase=None,
        extras=extras,
    )


def _start_pipeline_worker(
    session_id: str,
    principal: Any,
    principal_source: Any,
    dev_isolation_bypass: bool,
    run_cfg: dict[str, Any],
    h: dict[str, Any],
) -> None:
    requested_source = run_cfg["requested_source"]
    requested_origin = run_cfg["requested_origin"]
    force_parse_input_file = run_cfg["force_parse_input_file"]
    force_parse_format = run_cfg["force_parse_format"]
    manual_upload_rel = run_cfg["manual_upload_rel"]
    warehouse_override_url = run_cfg["warehouse_override_url"]
    direct_urls = run_cfg["direct_urls"]
    url_reference_hints = run_cfg["url_reference_hints"]

    output_bypass_flag = h["is_output_bypassed"](session_id)
    h["lock_session"](session_id)

    h["session_manager"].set_manual_source(session_id, requested_source, origin=requested_origin)

    h["logger"].info(
        {
            "level": "INFO",
            "type": "status",
            "message": "Parser connected. Starting parser run...",
            "session_id": session_id,
        }
    )
    h["logger"].info(
        {
            "level": "INFO",
            "type": "status",
            "message": f"Launching parser (source={requested_source}, output_bypass={'on' if output_bypass_flag else 'off'})",
            "session_id": session_id,
        }
    )

    run_id = f"run_{int(h['time'].time() * 1000)}"
    start_ts = h["datetime"].now(h["timezone"].utc).isoformat()
    h["log_run_event"](
        {
            "type": "start",
            "run_id": run_id,
            "session_id": session_id,
            "ts": start_ts,
            "source": requested_source,
            "output_bypass": output_bypass_flag,
            "status": "running",
            "manual_upload": manual_upload_rel,
            "direct_url_count": len(direct_urls),
            "warehouse_override_url": warehouse_override_url or None,
        }
    )

    cancel_flag = h["cancellation_manager"].get_flag(session_id)
    prompt_queue = h["get_prompt_queue"](session_id)

    def worker_wrapper():
        start_time = h["time"].time()
        output_dir = h["path_cls"](h["output_dir"])
        download_ready_emitted = h["threading"].Event()
        watcher_stop = h["threading"].Event()

        artifacts_before = _snapshot_output_artifacts(output_dir)

        def _output_watcher() -> None:
            while not watcher_stop.is_set() and not download_ready_emitted.is_set():
                h["time"].sleep(2)
                artifacts_after = _snapshot_output_artifacts(output_dir)
                new_artifacts = _detect_new_artifacts(artifacts_before, artifacts_after, start_time)
                if not new_artifacts:
                    continue
                newest_rel = max(new_artifacts, key=lambda rel: artifacts_after.get(rel, 0.0))
                if _emit_download_ready_for_rel(session_id, output_dir, newest_rel, h):
                    download_ready_emitted.set()

        watcher_thread = h["threading"].Thread(target=_output_watcher, daemon=True)
        watcher_thread.start()
        h["session_manager"].bind_thread_id(h["threading"].get_ident(), session_id)
        status = "error"
        err = None
        try:
            h["process_urls_for_web"](
                prompt_queue,
                session_id,
                cancel_flag,
                emit_func=h["socketio_emit_func"],
                output_bypass=output_bypass_flag,
                manual_source=requested_source,
                disable_internal_heartbeat=True,
                force_parse_input_file=force_parse_input_file,
                force_parse_format=force_parse_format,
                urls=direct_urls if direct_urls else None,
                url_reference_hints=url_reference_hints if direct_urls else None,
                warehouse_override_url=warehouse_override_url or None,
                principal=principal,
                principal_source=principal_source,
                dev_isolation_bypass=dev_isolation_bypass,
            )
            h["logger"].info(
                {
                    "level": "INFO",
                    "type": "status",
                    "message": "Parser run completed.",
                    "session_id": session_id,
                }
            )
            try:
                artifacts_after = _snapshot_output_artifacts(output_dir)
                new_artifacts = _detect_new_artifacts(artifacts_before, artifacts_after, start_time)
                if new_artifacts and not download_ready_emitted.is_set():
                    newest_rel = max(new_artifacts, key=lambda rel: artifacts_after.get(rel, 0.0))
                    if _emit_download_ready_for_rel(session_id, output_dir, newest_rel, h):
                        download_ready_emitted.set()
            except Exception:
                pass
            status = "ok"
            err = None
        except Exception as exc:
            h["logger"].error(
                {
                    "level": "ERROR",
                    "type": "exception",
                    "message": f"Parser run failed: {exc}",
                    "session_id": session_id,
                }
            )
            status = "error"
            err = str(exc)
        finally:
            watcher_stop.set()
            _finalize_worker_session(
                session_id,
                status,
                err,
                run_id,
                requested_source,
                requested_origin,
                output_bypass_flag,
                manual_upload_rel,
                direct_urls,
                warehouse_override_url,
                start_time,
                h,
            )
        h["session_manager"].unbind_thread_id(h["threading"].get_ident())

    thread = h["socketio"].start_background_task(worker_wrapper)
    h["session_manager"].set_thread(session_id, thread)


def run_ballot_lens_socket_handler(data=None, *, hooks: dict[str, Any]) -> None:
    payload = _normalize_payload(data)
    init_ctx = _initialize_session_and_auth(payload, hooks)
    if not init_ctx:
        return

    session_id = init_ctx["session_id"]
    dev_isolation_bypass = init_ctx["dev_isolation_bypass"]
    principal = init_ctx["principal"]
    principal_source = init_ctx["principal_source"]

    run_cfg = _prepare_run_inputs(payload, session_id, dev_isolation_bypass, hooks)
    _configure_logging_and_prompt(session_id, hooks)
    _start_pipeline_worker(
        session_id,
        principal,
        principal_source,
        dev_isolation_bypass,
        run_cfg,
        hooks,
    )
