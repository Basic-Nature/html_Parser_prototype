import type {
  RunEvent,
  RunMode,
  RunState,
} from '../contracts/runtime';

export interface SessionObservation {
  readonly index: number;
  readonly eventType: RunEvent['type'];
  readonly status: RunState['status'];
  readonly connectionState: RunState['context']['connectionState'];
  readonly checkpointId: string | null;
  readonly checkpointSequence: number | null;
  readonly actionRequired: string | null;
  readonly outputCount: number;
}

export interface SessionHistoryEntry {
  readonly sessionId: string;
  readonly runMode: RunMode | null;
  readonly sourceLabel: string | null;
  readonly status: RunState['status'];
  readonly connectionState: RunState['context']['connectionState'];
  readonly checkpointCount: number;
  readonly outputCount: number;
  readonly observations: readonly SessionObservation[];
}

export type SessionHistory = readonly SessionHistoryEntry[];

export const EMPTY_SESSION_HISTORY: SessionHistory = Object.freeze([]);

export function getSessionHistoryEntry(
  history: SessionHistory,
  sessionId: string | null,
): SessionHistoryEntry | null {
  if (!sessionId) {
    return null;
  }
  return history.find((entry) => entry.sessionId === sessionId) ?? null;
}

export function captureOwnedSession(
  history: SessionHistory,
  state: RunState,
  event: RunEvent,
): SessionHistory {
  const sessionId = state.context.sessionId;
  if (!sessionId) {
    return history;
  }

  const existingIndex = history.findIndex(
    (entry) => entry.sessionId === sessionId,
  );
  const existing = existingIndex >= 0 ? history[existingIndex] : null;
  const currentCheckpoint = state.context.currentCheckpoint
    ? state.context.checkpoints.find(
        (checkpoint) => checkpoint.checkpointId === state.context.currentCheckpoint,
      ) ?? null
    : null;

  const observation: SessionObservation = Object.freeze({
    index: (existing?.observations.length ?? 0) + 1,
    eventType: event.type,
    status: state.status,
    connectionState: state.context.connectionState,
    checkpointId: currentCheckpoint?.checkpointId ?? null,
    checkpointSequence: currentCheckpoint?.sequence ?? null,
    actionRequired: state.context.actionRequired?.summary ?? null,
    outputCount: state.context.outputs.length,
  });

  const nextEntry: SessionHistoryEntry = Object.freeze({
    sessionId,
    runMode: state.context.runMode,
    sourceLabel: state.context.sourceSummary?.displayLabel ?? null,
    status: state.status,
    connectionState: state.context.connectionState,
    checkpointCount: state.context.checkpoints.length,
    outputCount: state.context.outputs.length,
    observations: Object.freeze([
      ...(existing?.observations ?? []),
      observation,
    ]),
  });

  if (existingIndex < 0) {
    return Object.freeze([...history, nextEntry]);
  }

  return Object.freeze(
    history.map((entry, index) => (
      index === existingIndex ? nextEntry : entry
    )),
  );
}
