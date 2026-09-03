from pathlib import Path
from webapp.parser.services.ballot_lens_checkpoint_runtime import activate_ballot_lens_checkpoint_runtime,current_ballot_lens_checkpoint_runtime
from webapp.parser.services.trusted_ballot_lens_runtime import TrustedBallotLensRuntime
ROOT=Path('.')
def read(p):return (ROOT/p).read_text(encoding='utf-8-sig')
def test_f2h_trusted_runtime_boundary():
 events=[];rt=TrustedBallotLensRuntime('trusted_url','sess-owned',events.append)
 assert current_ballot_lens_checkpoint_runtime() is None
 with activate_ballot_lens_checkpoint_runtime(rt):assert current_ballot_lens_checkpoint_runtime() is rt
 assert current_ballot_lens_checkpoint_runtime() is None
 a=rt.record_checkpoint(checkpoint_id='source.resolve',state='complete',evidence_count=1);b=rt.record_checkpoint(checkpoint_id='source.acquire',state='complete',evidence_count=1);assert (a['sequence'],b['sequence'])==(1,2)
def test_f2h_existing_security_authority_preserved():
 s=read('webapp/parser/socket_ballot_lens_orchestration.py');assert 'require_cert_for_socket_action"]("ballot_lens"' in s;assert 'is_path_within_root(' in s;assert 'is_parser_eligible_url(' in s;assert 'guarded_ingestion_allowed"]("direct_urls")' in s;assert '_start_pipeline_worker(' in s
def test_f2h_public_authority_remains_public_only():
 s=read('webapp/parser/services/public_ballot_lens_runtime.py');assert 'PublicBrowserEgressGuard' in s and 'PublicRunMemoryState' in s and 'TrustedBallotLensRuntime' not in s
def test_f2h_shared_machine_and_command_reused():
 r=read('webapp/frontend/ballot-lens/contracts/runtime.ts');m=read('webapp/frontend/ballot-lens/state/runMachine.ts');t=read('webapp/frontend/ballot-lens/services/trustedExecution.ts');assert all(x in r for x in ("'trusted_url'","'manual_upload'","'worklist'"));assert 'incoming.sequence <= current.sequence' in m;assert "socket.emit('ballot_lens'" in t
