"""Structured evidence adapter for already-authorized trusted Ballot Lens runs."""
from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass,field
from datetime import datetime,timezone
import os
from .ballot_lens_checkpoint_runtime import CHECKPOINTS,STATES,ACTIONS
MODES=frozenset({"trusted_url","manual_upload","worklist"})
@dataclass
class TrustedBallotLensRuntime:
 run_mode:str;session_id:str;safe_emit:Callable[[dict[str,object]],None]|None=None
 _checkpoint_sequence:int=field(default=0,init=False)
 def __post_init__(self):
  self.run_mode=str(self.run_mode or "").strip();self.session_id=str(self.session_id or "").strip()
  if self.run_mode not in MODES or not self.session_id:raise ValueError("invalid trusted runtime")
 def _emit(self,p):
  if self.safe_emit:self.safe_emit({**p,"session_id":self.session_id,"run_mode":self.run_mode})
 def _text(self,v,maxlen,required=False):
  if v is None:
   if required:raise ValueError("required text")
   return None
  t=str(v).strip()
  if not t and required:raise ValueError("required text")
  return t[:maxlen] if t else None
 def emit_started(self):self._emit({"level":"INFO","type":"trusted_runtime","reason_code":"trusted_runtime_started","message":"Trusted parser runtime started."})
 def record_checkpoint(self,*,checkpoint_id,state,reason_code=None,summary=None,evidence_count=0,requires_action=False,action_type=None):
  if checkpoint_id not in CHECKPOINTS or state not in STATES:raise ValueError("invalid checkpoint")
  if isinstance(evidence_count,bool) or not isinstance(evidence_count,int) or evidence_count<0:raise ValueError("invalid evidence count")
  if action_type is not None and action_type not in ACTIONS:raise ValueError("invalid action")
  if requires_action!=(action_type is not None):raise ValueError("action fields disagree")
  self._checkpoint_sequence+=1
  cp={"checkpoint_id":checkpoint_id,"sequence":self._checkpoint_sequence,"state":state,"label":CHECKPOINTS[checkpoint_id],"reason_code":self._text(reason_code,128),"summary":self._text(summary,360),"evidence_count":evidence_count,"requires_action":requires_action,"action_type":action_type,"updated_at":datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
  level="ERROR" if state=="error" else "WARNING" if state=="warning" else "INFO"
  self._emit({"level":level,"type":"trusted_runtime_checkpoint","reason_code":"trusted_runtime_checkpoint_updated","message":"Structured trusted parser checkpoint updated.","checkpoint":dict(cp)});return cp
 def record_action_required(self,*,prompt_id,checkpoint_id,action_type,summary):
  if checkpoint_id not in CHECKPOINTS or action_type not in ACTIONS:raise ValueError("invalid action")
  action={"prompt_id":self._text(prompt_id,128,True),"checkpoint_id":checkpoint_id,"action_type":action_type,"summary":self._text(summary,360,True)}
  self._emit({"level":"WARNING","type":"trusted_runtime_action_required","reason_code":"trusted_runtime_action_required","message":"Structured trusted parser action is required.","action":action});return action
 def record_action_resolved(self,*,prompt_id):self._emit({"level":"INFO","type":"trusted_runtime_action_resolved","reason_code":"trusted_runtime_action_resolved","message":"Structured trusted parser action resolved.","prompt_id":self._text(prompt_id,128,True)})
 def record_result_checkpoints(self,*,headers:Sequence[str],contest):
  present=bool(str(contest or "").strip());self.record_checkpoint(checkpoint_id="contest.select",state="complete" if present else "warning",reason_code="trusted_contest_context_present" if present else "trusted_contest_context_missing",summary="Parser result returned contest context." if present else "Parser result did not expose contest context.",evidence_count=1 if present else 0)
  methods=[h for h in headers if isinstance(h,str) and " - " in h and not h.endswith(" - Total Votes") and not h.endswith(" - Total")]
  self.record_checkpoint(checkpoint_id="vote_methods.detect",state="complete" if methods else "warning",reason_code="trusted_vote_method_columns_present" if methods else "trusted_vote_method_columns_not_observed",summary="Method-specific result columns were observed." if methods else "No method-specific result columns were observed.",evidence_count=len(methods))
 def persisted_outputs(self,paths:Sequence[str]):
  out=[]
  for i,raw in enumerate(paths):
   rel=str(raw or "").replace("\\","/").strip("/")
   if rel:out.append({"output_id":f"{self.session_id}:persisted:{i+1}","label":os.path.basename(rel) or rel,"persistence":"persisted","download_available":True})
  return out
 def result_payload(self,*,terminal_status,terminal_reason_code,outputs):
  if terminal_status not in {"success","completed_with_errors","failed","cancelled"}:raise ValueError("invalid terminal status")
  return {"contract":"ballot_lens_trusted_runtime_result_v1","session_id":self.session_id,"run_mode":self.run_mode,"terminal_status":terminal_status,"terminal_reason_code":self._text(terminal_reason_code,128),"status_counts":{terminal_status:1},"outputs":list(outputs)}
