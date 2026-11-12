"""Focused shadow qualification for owned public terminal failure reporting."""
import ast
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get('W24_FIX1_CANDIDATE_ROOT', str(Path(__file__).resolve().parents[2])))
SERVER = ROOT / 'webapp/parser/socket_ballot_lens_orchestration.py'
CLIENT = ROOT / 'webapp/frontend/ballot-lens/components/workspace/WorkspaceShell.tsx'


def helper():
    tree = ast.parse(SERVER.read_text(encoding='utf-8'))
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == '_emit_public_registry_failure_terminal']
    assert len(funcs) == 1
    namespace = {'Any': Any}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SERVER), 'exec'), namespace)
    return namespace['_emit_public_registry_failure_terminal']


class FakeSocket:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def emit(self, event, body, *, room):
        if self.fail:
            raise RuntimeError('DO_NOT_ECHO_RAW_URL_OR_SECRET')
        self.events.append((event, body, room))


class FakeLogger:
    def __init__(self):
        self.items = []

    def warning(self, value):
        self.items.append(value)


def projection():
    return {'contest': 'President', 'format': 'HTML', 'registry_category': 'curated',
            'registry_source_id': 'blsrc_v1_' + 'a'*64, 'scope': 'statewide',
            'state': 'Utah', 'year': '2024'}


def test_failure_terminal_has_exact_safe_contract_no_rows_or_download():
    sock, log = FakeSocket(), FakeLogger()
    helper()({'socketio': sock, 'logger': log}, session_id='sess-owned', source_projection=projection())
    assert len(sock.events) == 1
    event, result, room = sock.events[0]
    assert (event, room) == ('public_registry_result', 'sess-owned')
    assert set(result) == {'contract', 'registry_source_id', 'source', 'outputs',
                           'status_counts', 'terminal_status', 'terminal_reason_code',
                           'download_available', 'persistent_output'}
    assert result['contract'] == 'ballot_lens_public_runtime_result_v1'
    assert result['source'] == projection()
    assert result['registry_source_id'] == projection()['registry_source_id']
    assert result['status_counts'] == {'error': 1}
    assert result['terminal_status'] == 'error'
    assert result['terminal_reason_code'] is None
    assert result['outputs'] == []
    assert result['download_available'] is False
    assert result['persistent_output'] is False
    assert 'url' not in str(result).lower()
    assert not log.items


def test_denies_untrusted_metadata_or_missing_session():
    s, log = FakeSocket(), FakeLogger()
    bad = projection()
    bad['url'] = 'https://REDACTED.EXAMPLE/source'
    fn = helper()
    fn({'socketio': s, 'logger': log}, session_id='sess-owned', source_projection=bad)
    fn({'socketio': s, 'logger': log}, session_id='', source_projection=projection())
    assert not s.events


def test_emit_failure_is_fail_closed_and_sanitized():
    s, log = FakeSocket(fail=True), FakeLogger()
    helper()({'socketio': s, 'logger': log}, session_id='sess-owned', source_projection=projection())
    assert len(log.items) == 1
    assert log.items[0]['reason_code'] == 'public_registry_failure_terminal_emit_unavailable'
    assert 'DO_NOT_ECHO' not in str(log.items)


def test_worker_exception_only_emits_failure_after_status_attempt():
    text = SERVER.read_text(encoding='utf-8')
    start = text.index('def _start_public_registry_runtime(')
    end = text.index('def _initialize_public_registry_authority(', start)
    runtime = text[start:end]
    exception_blocks = list(re.finditer(r'(?m)^        except Exception:$', runtime))
    assert len(exception_blocks) == 1
    p = exception_blocks[0].start()
    f = runtime.index('        finally:', p)
    branch = runtime[p:f]
    assert 'public_registry_runtime_failed' in branch
    assert '_emit_public_registry_failure_terminal(' in branch
    assert 'publish_from_public_result' not in branch
    assert 'public_registry_csv_ready' not in branch
    assert 'except PublicRunAdmissionError:' in runtime


def test_frontend_distinguishes_failed_terminal_from_successful_preview():
    ts = CLIENT.read_text(encoding='utf-8')
    assert "runState.context.terminalStatus === 'failed'" in ts
    assert "const publicRuntimeFailed = runtimeResult !== null" in ts
    assert '{runtimeResult && publicRuntimeFailed ? (' in ts
    assert 'Approved-source parsing did not complete' in ts
    assert 'No validated precinct preview or CSV download is available' in ts
    assert ts.index('{runtimeResult && publicRuntimeFailed ? (') < ts.index('Safe parser result is ready')
