import { fromTransition } from 'xstate';

import {
  createInitialCheckpoints,
  isCheckpointId,
  type ParserCheckpoint,
} from '../contracts/checkpoints';
import type {
  ConnectionState,
  RunContext,
  RunEvent,
  RunOutput,
  RunState,
  ResumableRunStatus,
} from '../contracts/runtime';

const activeStatuses = new Set<RunState['status']>([
  'submitting',
  'awaiting_session',
  'running',
]);

function createInitialRunContext(
  connectionState: ConnectionState = 'not_connected',
): RunContext {
  return {
    runMode: null,
    sessionId: null,
    sourceSummary: null,
    provider: null,
    checkpoints: createInitialCheckpoints(),
    currentCheckpoint: null,
    actionRequired: null,
    terminalStatus: null,
    terminalReasonCode: null,
    statusCounts: {},
    outputs: [],
    startedAt: null,
    completedAt: null,
    connectionState,
  };
}

export function createInitialRunState(
  connectionState: ConnectionState = 'not_connected',
): RunState {
  return {
    status: 'idle',
    resumeStatus: null,
    context: createInitialRunContext(connectionState),
  };
}

function ownsSession(state: RunState, sessionId: string): boolean {
  return (
    sessionId.length > 0
    && state.context.sessionId !== null
    && state.context.sessionId === sessionId
  );
}

function isValidSourceSelection(event: Extract<RunEvent, {
  type: 'SOURCE_SELECTED';
}>): boolean {
  if (event.sourceSummary.runMode !== event.runMode) {
    return false;
  }

  if (
    event.runMode === 'public_registry'
    && !event.sourceSummary.registrySourceId
  ) {
    return false;
  }

  if (
    event.runMode !== 'public_registry'
    && event.sourceSummary.registrySourceId
  ) {
    return false;
  }

  return event.sourceSummary.displayLabel.trim().length > 0;
}

function isValidOutput(output: RunOutput): boolean {
  return !(
    output.persistence === 'memory_only'
    && output.downloadAvailable !== false
  );
}

function isValidStatusCounts(
  statusCounts: Readonly<Record<string, number>>,
): boolean {
  return Object.values(statusCounts).every(
    (value) => Number.isInteger(value) && value >= 0,
  );
}

function replaceCheckpoint(
  checkpoints: readonly ParserCheckpoint[],
  incoming: ParserCheckpoint,
): readonly ParserCheckpoint[] | null {
  if (!isCheckpointId(incoming.checkpointId)) {
    return null;
  }

  const index = checkpoints.findIndex(
    ({ checkpointId }) => checkpointId === incoming.checkpointId,
  );
  if (index < 0) {
    return null;
  }

  const current = checkpoints[index];
  if (!current || incoming.sequence <= current.sequence) {
    return null;
  }

  const updated = checkpoints.slice();
  updated[index] = incoming;
  return updated;
}

function resumeStatusFor(state: RunState): ResumableRunStatus | null {
  if (
    state.status === 'submitting'
    || state.status === 'awaiting_session'
    || state.status === 'running'
  ) {
    return state.status;
  }
  return state.resumeStatus;
}

function restoredStatus(state: RunState): RunState['status'] {
  if (state.resumeStatus) {
    return state.resumeStatus;
  }
  if (state.context.sessionId) {
    return 'running';
  }
  if (state.context.runMode) {
    return 'source_selected';
  }
  return 'idle';
}

export function reduceRunState(
  state: RunState,
  event: RunEvent,
): RunState {
  switch (event.type) {
    case 'SOURCE_SELECTED': {
      if (
        !['idle', 'source_selected', 'terminal'].includes(state.status)
        || !isValidSourceSelection(event)
      ) {
        return state;
      }

      return {
        status: 'source_selected',
        resumeStatus: null,
        context: {
          ...createInitialRunContext(state.context.connectionState),
          runMode: event.runMode,
          sourceSummary: event.sourceSummary,
        },
      };
    }

    case 'SUBMIT_REQUESTED':
      if (state.status !== 'source_selected') {
        return state;
      }
      return { ...state, status: 'submitting' };

    case 'SUBMISSION_ACCEPTED':
      if (state.status !== 'submitting') {
        return state;
      }
      return { ...state, status: 'awaiting_session' };

    case 'SESSION_CORRELATED': {
      if (
        !['submitting', 'awaiting_session'].includes(state.status)
        || state.context.runMode !== event.runMode
        || state.context.sessionId !== null
        || event.sessionId.trim().length === 0
      ) {
        return state;
      }

      return {
        status: 'running',
        resumeStatus: null,
        context: {
          ...state.context,
          sessionId: event.sessionId,
          startedAt: event.startedAt,
        },
      };
    }

    case 'PROVIDER_IDENTIFIED':
      if (
        state.status !== 'running'
        || !ownsSession(state, event.sessionId)
        || event.provider.trim().length === 0
      ) {
        return state;
      }
      return {
        ...state,
        context: {
          ...state.context,
          provider: event.provider,
        },
      };

    case 'CHECKPOINT_UPDATED': {
      if (
        state.status !== 'running'
        || !ownsSession(state, event.sessionId)
      ) {
        return state;
      }

      const checkpoints = replaceCheckpoint(
        state.context.checkpoints,
        event.checkpoint,
      );
      if (!checkpoints) {
        return state;
      }

      return {
        ...state,
        context: {
          ...state.context,
          checkpoints,
          currentCheckpoint: event.checkpoint.checkpointId,
        },
      };
    }

    case 'ACTION_REQUIRED':
      if (
        state.status !== 'running'
        || !ownsSession(state, event.sessionId)
        || !isCheckpointId(event.action.checkpointId)
        || event.action.promptId.trim().length === 0
      ) {
        return state;
      }
      return {
        ...state,
        context: {
          ...state.context,
          actionRequired: event.action,
          currentCheckpoint: event.action.checkpointId,
        },
      };

    case 'ACTION_RESOLVED':
      if (
        state.status !== 'running'
        || !ownsSession(state, event.sessionId)
        || state.context.actionRequired?.promptId !== event.promptId
      ) {
        return state;
      }
      return {
        ...state,
        context: {
          ...state.context,
          actionRequired: null,
        },
      };

    case 'RUN_TERMINATED':
      if (
        !['running', 'disconnected'].includes(state.status)
        || !ownsSession(state, event.sessionId)
        || !event.outputs.every(isValidOutput)
        || !isValidStatusCounts(event.statusCounts)
      ) {
        return state;
      }
      return {
        status: 'terminal',
        resumeStatus: null,
        context: {
          ...state.context,
          terminalStatus: event.terminalStatus,
          terminalReasonCode: event.terminalReasonCode,
          statusCounts: event.statusCounts,
          outputs: event.outputs,
          completedAt: event.completedAt,
        },
      };

    case 'CONNECTION_ESTABLISHED':
      if (state.status === 'disconnected') {
        return {
          ...state,
          status: restoredStatus(state),
          resumeStatus: null,
          context: {
            ...state.context,
            connectionState: 'connected',
          },
        };
      }
      return {
        ...state,
        context: {
          ...state.context,
          connectionState: 'connected',
        },
      };

    case 'CONNECTION_LOST': {
      const nextConnectionState: ConnectionState = 'disconnected';
      if (activeStatuses.has(state.status)) {
        return {
          ...state,
          status: 'disconnected',
          resumeStatus: resumeStatusFor(state),
          context: {
            ...state.context,
            connectionState: nextConnectionState,
          },
        };
      }
      return {
        ...state,
        context: {
          ...state.context,
          connectionState: nextConnectionState,
        },
      };
    }

    case 'CONNECTION_RESTORED':
      if (state.status !== 'disconnected') {
        return state;
      }
      return {
        ...state,
        status: restoredStatus(state),
        resumeStatus: null,
        context: {
          ...state.context,
          connectionState: 'connected',
        },
      };

    case 'RESET':
      return createInitialRunState(state.context.connectionState);

    default:
      return state;
  }
}

export const runActorLogic = fromTransition(
  reduceRunState,
  createInitialRunState(),
);
