import type {
  ParserCheckpoint,
} from '../contracts/checkpoints';
import type { RunState } from '../contracts/runtime';

export function ownsActiveSession(
  state: RunState,
  sessionId: string,
): boolean {
  return (
    sessionId.length > 0
    && state.context.sessionId !== null
    && state.context.sessionId === sessionId
  );
}

export function canSubmit(state: RunState): boolean {
  return state.status === 'source_selected';
}

export function hasUnresolvedAction(state: RunState): boolean {
  return state.context.actionRequired !== null;
}

export function isTerminal(state: RunState): boolean {
  return state.status === 'terminal';
}

export function selectCurrentCheckpoint(
  state: RunState,
): ParserCheckpoint | null {
  const currentId = state.context.currentCheckpoint;
  if (!currentId) {
    return null;
  }

  return (
    state.context.checkpoints.find(
      ({ checkpointId }) => checkpointId === currentId,
    ) ?? null
  );
}
