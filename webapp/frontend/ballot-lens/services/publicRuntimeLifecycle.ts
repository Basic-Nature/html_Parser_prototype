import {
  parsePublicRuntimeResult,
  PUBLIC_EXECUTION_CONTEXT_CONTRACT,
  PUBLIC_MEMORY_PREVIEW_CONTRACT,
  type PublicRuntimeResult,
} from '../contracts/publicRuntime';
import { PublicCompactCoordinator, COMPACT_WIRE_EVENTS } from './publicCompactCoordinator';
import type { CompactSummary } from './publicCompactAssembler';
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
const ALL_PUBLIC_EVENTS = Object.freeze([...PUBLIC_SOCKET_OBSERVATION_EVENTS, ...COMPACT_WIRE_EVENTS]);
type OwnedPublicEventName = typeof ALL_PUBLIC_EVENTS[number];

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
  readonly nowSeconds?: () => number;
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
  // The backend emits a compact-ready result before the first page. No terminal
  // state or rows may be exposed until the final server acknowledgment.
  if (result.compact_delivery) return null;
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

function compactToRuntimeResult(pending: PublicRuntimeResult, summary: CompactSummary): PublicRuntimeResult {
  if (!pending.compact_delivery || pending.registry_source_id !== summary.scope.registry_source_id ||
      pending.source.state !== summary.scope.state || pending.source.year !== summary.scope.year ||
      pending.source.contest !== summary.scope.contest ||
      pending.compact_delivery.rows !== summary.rowCount ||
      pending.compact_delivery.page_count !== summary.pageCount) {
    throw new Error('Compact release differs from the owned runtime result');
  }
  return parsePublicRuntimeResult({
    contract: pending.contract,
    registry_source_id: pending.registry_source_id,
    source: pending.source,
    outputs: [{
      contract: PUBLIC_MEMORY_PREVIEW_CONTRACT,
      registry_source_id: pending.registry_source_id,
      source: pending.source,
      headers: summary.headers,
      rows: summary.rows,
      row_count: summary.rowCount,
      output_mode: 'MEMORY_PREVIEW_ONLY',
      download_available: false,
      persistent_output: false,
      execution_context_contract: PUBLIC_EXECUTION_CONTEXT_CONTRACT,
    }],
    status_counts: pending.status_counts,
    terminal_status: pending.terminal_status,
    terminal_reason_code: pending.terminal_reason_code,
    download_available: false,
    persistent_output: false,
  });
}

export function installPublicRuntimeLifecycle(
  socket: DormantBallotLensSocket,
  options: PublicRuntimeLifecycleOptions,
): () => void {
  const now = options.now ?? (() => new Date().toISOString());
  const nowSeconds = options.nowSeconds ?? (() => Math.floor(Date.now() / 1000));
  const listeners = new Map<OwnedPublicEventName, (payload?: unknown) => void>();
  let pendingResult: PublicRuntimeResult | null = null;
  const coordinator = new PublicCompactCoordinator(
    command => { socket.emit(command.event, command.payload); },
    nowSeconds,
  );
  const owned = (): boolean => {
    const state = options.getRunState();
    return state.status === 'running' && state.context.sessionId !== null &&
      ownsSelectedPublicRun(state, options.getSelectedRegistrySourceId());
  };
  const discard = (): void => { pendingResult = null; coordinator.discard(); };

  for (const eventName of ALL_PUBLIC_EVENTS) {
    const listener = (payload?: unknown) => {
      try {
        if (eventName === 'disconnect') discard();
        if (eventName === 'public_registry_result') {
          const observation = normalizePublicSocketObservation(eventName, payload);
          if (observation?.kind === 'runtime_result' && observation.result.compact_delivery) {
            if (!owned() || observation.result.registry_source_id !== options.getSelectedRegistrySourceId()) return;
            coordinator.start({
              sourceId: observation.result.registry_source_id,
              rows: observation.result.compact_delivery.rows,
              pageCount: observation.result.compact_delivery.page_count,
            });
            pendingResult = observation.result;
            return;
          }
        }
        if ((COMPACT_WIRE_EVENTS as readonly string[]).includes(eventName)) {
          if (!owned() || !pendingResult) return;
          const summary = coordinator.handle(
            eventName as typeof COMPACT_WIRE_EVENTS[number], payload,
            options.getSelectedRegistrySourceId() ?? '',
          );
          if (!summary) return;
          const result = compactToRuntimeResult(pendingResult, summary);
          const decision = routePublicSocketObservation(
            'public_registry_result', result, options.getRunState(),
            options.getSelectedRegistrySourceId(), now(),
          );
          discard();
          if (!decision || !decision.runtimeResult) throw new Error('Compact result has no owned terminal decision');
          options.dispatch(decision.event);
          options.onRuntimeResult(decision.runtimeResult);
          return;
        }
        if (eventName === 'public_registry_result' && pendingResult) {
          throw new Error('Legacy result received during compact delivery');
        }
        const decision = routePublicSocketObservation(
          eventName as PublicSocketEventName,
          payload, options.getRunState(),
          options.getSelectedRegistrySourceId(), now(),
        );
        if (!decision) return;
        options.dispatch(decision.event);
        if (decision.runtimeResult) options.onRuntimeResult(decision.runtimeResult);
      } catch {
        discard();
        options.onProtocolError();
      }
    };
    listeners.set(eventName, listener);
    socket.on(eventName, listener);
  }

  return () => {
    discard();
    for (const [eventName, listener] of listeners) socket.off(eventName, listener);
  };
}
