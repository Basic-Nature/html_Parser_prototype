import type {
  PublicRuntimeResult,
} from '../contracts/publicRuntime';
import type {
  RunEvent,
  RunOutput,
  RunState,
  TerminalStatus,
} from '../contracts/runtime';
import type {
  DormantBallotLensSocket,
} from './socketClient';
import {
  normalizePublicSocketObservation,
  PUBLIC_SOCKET_OBSERVATION_EVENTS,
} from './socketAdapter';

type PublicSocketEventName =
  typeof PUBLIC_SOCKET_OBSERVATION_EVENTS[number];

export interface PublicRuntimeLifecycleDecision {
  readonly event: RunEvent;
  readonly runtimeResult?: PublicRuntimeResult;
}

export interface PublicRuntimeLifecycleOptions {
  readonly getRunState: () => RunState;
  readonly getSelectedRegistrySourceId: () => string | null;
  readonly dispatch: (event: RunEvent) => void;
  readonly onRuntimeResult: (result: PublicRuntimeResult) => void;
  readonly onProtocolError: () => void;
  readonly now?: () => string;
}

const FAILED_TERMINAL_STATUSES = new Set([
  'error',
  'fail',
  'quarantined',
  'rejected',
]);

const ERROR_STATUS_COUNTS = new Set([
  'error',
  'fail',
  'quarantined',
  'rejected',
]);

function hasErrorStatusCount(
  statusCounts: Readonly<Record<string, number>>,
): boolean {
  return Object.entries(statusCounts).some(
    ([status, count]) => ERROR_STATUS_COUNTS.has(status) && count > 0,
  );
}

function mapTerminalStatus(
  result: PublicRuntimeResult,
): TerminalStatus {
  const terminalStatus = result.terminal_status;
  if (terminalStatus === 'cancelled') return 'cancelled';
  if (FAILED_TERMINAL_STATUSES.has(terminalStatus ?? '')) return 'failed';
  if (
    terminalStatus === 'partial'
    || hasErrorStatusCount(result.status_counts)
  ) {
    return 'completed_with_errors';
  }
  if (
    terminalStatus === 'success'
    || terminalStatus === 'skipped_data_exists'
  ) {
    return 'success';
  }
  throw new Error('Unsupported public terminal status');
}

function summarizeOutputs(
  result: PublicRuntimeResult,
): readonly RunOutput[] {
  return Object.freeze(result.outputs.map((output, index) => Object.freeze({
    outputId: `${result.registry_source_id}:memory-preview:${index + 1}`,
    label: `Memory preview ${index + 1} (${output.row_count} rows)`,
    persistence: 'memory_only' as const,
    downloadAvailable: false as const,
  })));
}

function ownsSelectedPublicRun(
  state: RunState,
  selectedRegistrySourceId: string | null,
): boolean {
  return (
    selectedRegistrySourceId !== null
    && state.context.runMode === 'public_registry'
    && state.context.sourceSummary?.registrySourceId
      === selectedRegistrySourceId
  );
}


function ownsStructuredObservation(
  state: RunState,
  selectedRegistrySourceId: string | null,
  sessionId: string,
  registrySourceId: string,
): boolean {
  return (
    state.status === 'running'
    && state.context.sessionId === sessionId
    && ownsSelectedPublicRun(state, selectedRegistrySourceId)
    && registrySourceId === selectedRegistrySourceId
  );
}

export function routePublicSocketObservation(
  eventName: PublicSocketEventName,
  payload: unknown,
  state: RunState,
  selectedRegistrySourceId: string | null,
  observedAt: string,
): PublicRuntimeLifecycleDecision | null {
  const observation = normalizePublicSocketObservation(eventName, payload);
  if (!observation) return null;

  if (observation.kind === 'connection') {
    if (observation.state === 'disconnected') {
      return Object.freeze({ event: Object.freeze({
        type: 'CONNECTION_LOST' as const,
      }) });
    }
    return Object.freeze({ event: Object.freeze({
      type: state.status === 'disconnected'
        ? 'CONNECTION_RESTORED' as const
        : 'CONNECTION_ESTABLISHED' as const,
    }) });
  }

  if (
    observation.kind === 'runtime_started'
    && ['submitting', 'awaiting_session'].includes(state.status)
    && state.context.sessionId === null
    && ownsSelectedPublicRun(state, selectedRegistrySourceId)
  ) {
    return Object.freeze({ event: Object.freeze({
      type: 'SESSION_CORRELATED' as const,
      runMode: 'public_registry' as const,
      sessionId: observation.sessionId,
      startedAt: observedAt,
    }) });
  }


  if (observation.kind === 'checkpoint') {
    if (!ownsStructuredObservation(state, selectedRegistrySourceId, observation.sessionId, observation.registrySourceId)) return null;
    return Object.freeze({ event: Object.freeze({ type: 'CHECKPOINT_UPDATED' as const, sessionId: observation.sessionId, checkpoint: observation.checkpoint }) });
  }
  if (observation.kind === 'action_required') {
    if (!ownsStructuredObservation(state, selectedRegistrySourceId, observation.sessionId, observation.registrySourceId)) return null;
    return Object.freeze({ event: Object.freeze({ type: 'ACTION_REQUIRED' as const, sessionId: observation.sessionId, action: observation.action }) });
  }
  if (observation.kind === 'action_resolved') {
    if (!ownsStructuredObservation(state, selectedRegistrySourceId, observation.sessionId, observation.registrySourceId)) return null;
    return Object.freeze({ event: Object.freeze({ type: 'ACTION_RESOLVED' as const, sessionId: observation.sessionId, promptId: observation.promptId }) });
  }

  if (observation.kind !== 'runtime_result') return null;
  if (
    !['running', 'disconnected'].includes(state.status)
    || state.context.sessionId === null
    || !ownsSelectedPublicRun(state, selectedRegistrySourceId)
    || observation.result.registry_source_id !== selectedRegistrySourceId
  ) {
    return null;
  }

  const result = observation.result;
  return Object.freeze({
    event: Object.freeze({
      type: 'RUN_TERMINATED' as const,
      sessionId: state.context.sessionId,
      terminalStatus: mapTerminalStatus(result),
      terminalReasonCode: result.terminal_reason_code,
      statusCounts: result.status_counts,
      outputs: summarizeOutputs(result),
      completedAt: observedAt,
    }),
    runtimeResult: result,
  });
}

export function installPublicRuntimeLifecycle(
  socket: DormantBallotLensSocket,
  options: PublicRuntimeLifecycleOptions,
): () => void {
  const now = options.now ?? (() => new Date().toISOString());
  const listeners = new Map<
    PublicSocketEventName,
    (payload?: unknown) => void
  >();

  for (const eventName of PUBLIC_SOCKET_OBSERVATION_EVENTS) {
    const listener = (payload?: unknown) => {
      try {
        const decision = routePublicSocketObservation(
          eventName,
          payload,
          options.getRunState(),
          options.getSelectedRegistrySourceId(),
          now(),
        );
        if (!decision) return;
        options.dispatch(decision.event);
        if (decision.runtimeResult) {
          options.onRuntimeResult(decision.runtimeResult);
        }
      } catch {
        options.onProtocolError();
      }
    };
    listeners.set(eventName, listener);
    socket.on(eventName, listener);
  }

  return () => {
    for (const [eventName, listener] of listeners) {
      socket.off(eventName, listener);
    }
  };
}
