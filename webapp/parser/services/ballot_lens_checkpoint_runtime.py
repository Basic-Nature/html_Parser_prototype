"""Mode-neutral Ballot Lens structured-checkpoint context; no execution authority."""
from __future__ import annotations
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Protocol
CHECKPOINTS={"source.resolve":"Resolve Source","provider.detect":"Provider Detection","source.acquire":"Acquire","structure.detect":"Detect Structure","contest.select":"Contest Selection","vote_methods.detect":"Vote Method Selection","normalize.rows":"Normalize","validate.results":"Validate","preview.publish":"Preview"}
STATES=frozenset({"pending","active","complete","warning","error"})
ACTIONS=frozenset({"contest_selection","vote_method_selection","challenge","other"})
class CheckpointRuntime(Protocol):
 def record_checkpoint(self,*,checkpoint_id:str,state:str,reason_code:str|None=None,summary:str|None=None,evidence_count:int=0,requires_action:bool=False,action_type:str|None=None)->dict[str,object]:...
 def record_action_required(self,*,prompt_id:str,checkpoint_id:str,action_type:str,summary:str)->dict[str,object]:...
 def record_result_checkpoints(self,*,headers:Sequence[str],contest:object|None)->None:...
_ACTIVE:ContextVar[CheckpointRuntime|None]=ContextVar("electionpulse_active_ballot_lens_checkpoint_runtime",default=None)
def current_ballot_lens_checkpoint_runtime()->CheckpointRuntime|None:return _ACTIVE.get()
@contextmanager
def activate_ballot_lens_checkpoint_runtime(runtime:CheckpointRuntime)->Iterator[CheckpointRuntime]:
 if runtime is None:raise ValueError("checkpoint runtime required")
 if current_ballot_lens_checkpoint_runtime() is not None:raise RuntimeError("nested checkpoint runtime")
 token:Token[CheckpointRuntime|None]=_ACTIVE.set(runtime)
 try:yield runtime
 finally:_ACTIVE.reset(token)
