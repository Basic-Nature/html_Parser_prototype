import { describe, expect, it } from 'vitest';

import type { RunEvent, RunState } from '../contracts/runtime';
import {
  installPublicRuntimeLifecycle,
} from '../services/publicRuntimeLifecycle';
import type {
  DormantBallotLensSocket,
} from '../services/socketClient';
import {
  createInitialRunState,
  reduceRunState,
} from '../state/runMachine';

const source = Object.freeze({
  contest: 'President',
  format: 'HTML',
  registry_category: 'curated',
  registry_source_id: 'source-ut',
  scope: 'statewide',
  state: 'Utah',
  year: '2024',
});

const preview = Object.freeze({
  contract: 'ballot_lens_public_memory_preview_v1',
  registry_source_id: 'source-ut',
  source,
  headers: ['Precinct', 'Candidate - Total Votes'],
  rows: [{ Precinct: null, 'Candidate - Total Votes': 42 }],
  row_count: 1,
  output_mode: 'MEMORY_PREVIEW_ONLY',
  download_available: false,
  persistent_output: false,
  execution_context_contract: 'ballot_lens_public_execution_context_v1',
});

const result = Object.freeze({
  contract: 'ballot_lens_public_runtime_result_v1',
  registry_source_id: 'source-ut',
  source,
  outputs: [preview],
  status_counts: { success: 1 },
  terminal_status: 'success',
  terminal_reason_code: null,
  download_available: false,
  persistent_output: false,
});

function awaitingSession(): RunState {
  let state = createInitialRunState();
  state = reduceRunState(state, {
    type: 'SOURCE_SELECTED',
    runMode: 'public_registry',
    sourceSummary: {
      runMode: 'public_registry',
      displayLabel: 'Utah 2024 President',
      registrySourceId: 'source-ut',
    },
  });
  state = reduceRunState(state, { type: 'SUBMIT_REQUESTED' });
  return reduceRunState(state, { type: 'SUBMISSION_ACCEPTED' });
}

function socketHarness() {
  const listeners = new Map<string, (payload?: unknown) => void>();
  const onEvents: string[] = [];
  const offEvents: string[] = [];
  const socket: DormantBallotLensSocket = {
    connected: false,
    connect() { return this; },
    disconnect() { return this; },
    emit() { return this; },
    on(event, listener) {
      onEvents.push(event);
      listeners.set(event, listener);
      return this;
    },
    off(event, listener) {
      offEvents.push(event);
      if (listeners.get(event) === listener) listeners.delete(event);
      return this;
    },
  };
  return {
    socket,
    onEvents,
    offEvents,
    listeners,
    trigger(event: string, payload?: unknown) {
      listeners.get(event)?.(payload);
    },
  };
}

describe('F2-E3/E4 public runtime lifecycle', () => {
  it('registers exact lifecycle listeners and removes each one', () => {
    const harness = socketHarness();
    const detach = installPublicRuntimeLifecycle(harness.socket, {
      getRunState: awaitingSession,
      getSelectedRegistrySourceId: () => 'source-ut',
      dispatch: () => undefined,
      onRuntimeResult: () => undefined,
      onProtocolError: () => undefined,
    });

    expect(harness.onEvents).toEqual([
      'connect',
      'disconnect',
      'parser_output',
      'public_registry_result',
    ]);
    expect(harness.socket.connected).toBe(false);

    detach();
    expect(harness.offEvents).toEqual(harness.onEvents);
    expect(harness.listeners.size).toBe(0);
  });

  it('owns the server session, reconnects, and retains the typed result', () => {
    const harness = socketHarness();
    let state = awaitingSession();
    const dispatched: RunEvent[] = [];
    const retainedResults: unknown[] = [];
    const protocolErrors: unknown[] = [];
    const dispatch = (event: RunEvent) => {
      dispatched.push(event);
      state = reduceRunState(state, event);
    };

    installPublicRuntimeLifecycle(harness.socket, {
      getRunState: () => state,
      getSelectedRegistrySourceId: () => 'source-ut',
      dispatch,
      onRuntimeResult: (runtimeResult) => retainedResults.push(runtimeResult),
      onProtocolError: () => protocolErrors.push('protocol-error'),
      now: () => '2026-09-02T08:30:00Z',
    });

    harness.trigger('parser_output', {
      reason_code: 'public_registry_runtime_started',
      session_id: 'server-session',
    });
    expect(state.status).toBe('running');
    expect(state.context.sessionId).toBe('server-session');

    harness.trigger('disconnect');
    expect(state.status).toBe('disconnected');
    expect(state.context.sessionId).toBe('server-session');

    harness.trigger('connect');
    expect(state.status).toBe('running');
    expect(state.context.sessionId).toBe('server-session');

    harness.trigger('public_registry_result', result);
    expect(state.status).toBe('terminal');
    expect(state.context.terminalStatus).toBe('success');
    expect(state.context.outputs).toEqual([{
      outputId: 'source-ut:memory-preview:1',
      label: 'Memory preview 1 (1 rows)',
      persistence: 'memory_only',
      downloadAvailable: false,
    }]);
    expect(retainedResults).toHaveLength(1);
    expect(retainedResults[0]).toMatchObject({
      outputs: [{ rows: [{ Precinct: null }] }],
    });
    expect(protocolErrors).toEqual([]);
    expect(dispatched.map(({ type }) => type)).toEqual([
      'SESSION_CORRELATED',
      'CONNECTION_LOST',
      'CONNECTION_RESTORED',
      'RUN_TERMINATED',
    ]);
  });

  it('rejects stale, foreign-source, and malformed terminal results', () => {
    const harness = socketHarness();
    let state = awaitingSession();
    const dispatched: RunEvent[] = [];
    let retainedCount = 0;
    let protocolErrorCount = 0;
    const dispatch = (event: RunEvent) => {
      dispatched.push(event);
      state = reduceRunState(state, event);
    };

    installPublicRuntimeLifecycle(harness.socket, {
      getRunState: () => state,
      getSelectedRegistrySourceId: () => 'source-ut',
      dispatch,
      onRuntimeResult: () => { retainedCount += 1; },
      onProtocolError: () => { protocolErrorCount += 1; },
      now: () => '2026-09-02T08:30:00Z',
    });

    harness.trigger('public_registry_result', result);
    expect(dispatched).toEqual([]);

    harness.trigger('parser_output', {
      reason_code: 'public_registry_runtime_started',
      session_id: 'owned-session',
    });
    harness.trigger('parser_output', {
      reason_code: 'public_registry_runtime_started',
      session_id: 'stale-session',
    });
    expect(state.context.sessionId).toBe('owned-session');

    harness.trigger('public_registry_result', {
      ...result,
      registry_source_id: 'source-other',
      source: { ...source, registry_source_id: 'source-other' },
      outputs: [],
    });
    expect(state.status).toBe('running');
    expect(retainedCount).toBe(0);

    harness.trigger('public_registry_result', {
      ...result,
      download_available: true,
    });
    expect(state.status).toBe('running');
    expect(retainedCount).toBe(0);
    expect(protocolErrorCount).toBe(1);

    harness.trigger('public_registry_result', {
      ...result,
      terminal_status: 'mystery',
    });
    expect(state.status).toBe('running');
    expect(retainedCount).toBe(0);
    expect(protocolErrorCount).toBe(2);
  });

  it('maps partial and failed public outcomes without inventing checkpoints', () => {
    const harness = socketHarness();
    let state = awaitingSession();
    const dispatch = (event: RunEvent) => {
      state = reduceRunState(state, event);
    };
    installPublicRuntimeLifecycle(harness.socket, {
      getRunState: () => state,
      getSelectedRegistrySourceId: () => 'source-ut',
      dispatch,
      onRuntimeResult: () => undefined,
      onProtocolError: () => undefined,
    });

    harness.trigger('parser_output', {
      reason_code: 'public_registry_runtime_started',
      session_id: 'owned-session',
    });
    harness.trigger('public_registry_result', {
      ...result,
      status_counts: { partial: 1 },
      terminal_status: 'partial',
    });

    expect(state.context.terminalStatus).toBe('completed_with_errors');
    expect(state.context.checkpoints.every(({ sequence }) => sequence === 0))
      .toBe(true);
  });

  it('accepts only owned structured checkpoint and action evidence', () => {
    const harness = socketHarness();
    let state = awaitingSession();
    const dispatched: RunEvent[] = [];
    let protocolErrorCount = 0;
    const dispatch = (event: RunEvent) => { dispatched.push(event); state = reduceRunState(state, event); };
    installPublicRuntimeLifecycle(harness.socket, {
      getRunState: () => state,
      getSelectedRegistrySourceId: () => 'source-ut', dispatch,
      onRuntimeResult: () => undefined,
      onProtocolError: () => { protocolErrorCount += 1; },
      now: () => '2026-09-02T08:30:00Z',
    });
    harness.trigger('parser_output', { reason_code: 'public_registry_runtime_started', session_id: 'owned-session' });
    harness.trigger('parser_output', {
      reason_code: 'public_registry_checkpoint_updated', session_id: 'owned-session', registry_source_id: 'source-ut',
      checkpoint: { checkpoint_id: 'source.resolve', sequence: 1, state: 'complete', label: 'Resolve Source', reason_code: 'approved_public_registry_source_resolved', summary: 'Approved registry source authority confirmed.', evidence_count: 1, requires_action: false, action_type: null, updated_at: '2026-09-02T08:30:01Z' },
    });
    expect(state.context.checkpoints[0]).toMatchObject({ checkpointId: 'source.resolve', sequence: 1, state: 'complete' });
    harness.trigger('parser_output', { level: 'INFO', type: 'status', message: 'pretend checkpoint complete', session_id: 'owned-session' });
    expect(state.context.checkpoints.filter(({ sequence }) => sequence > 0)).toHaveLength(1);
    harness.trigger('parser_output', {
      reason_code: 'public_registry_checkpoint_updated', session_id: 'foreign-session', registry_source_id: 'source-ut',
      checkpoint: { checkpoint_id: 'source.acquire', sequence: 2, state: 'complete', label: 'Acquire', reason_code: null, summary: 'Foreign event.', evidence_count: 1, requires_action: false, action_type: null, updated_at: '2026-09-02T08:30:02Z' },
    });
    expect(state.context.currentCheckpoint).toBe('source.resolve');
    harness.trigger('parser_output', {
      reason_code: 'public_registry_action_required', session_id: 'owned-session', registry_source_id: 'source-ut',
      action: { prompt_id: 'public-challenge-assist', checkpoint_id: 'source.acquire', action_type: 'challenge', summary: 'Browser challenge requires interaction unavailable in public mode.' },
    });
    expect(state.context.actionRequired).toEqual({ promptId: 'public-challenge-assist', checkpointId: 'source.acquire', actionType: 'challenge', summary: 'Browser challenge requires interaction unavailable in public mode.' });
    expect(protocolErrorCount).toBe(0);
    expect(dispatched.map(({ type }) => type)).toContain('CHECKPOINT_UPDATED');
    expect(dispatched.map(({ type }) => type)).toContain('ACTION_REQUIRED');
  });

  it('rejects malformed or unknown structured checkpoint evidence', () => {
    const harness = socketHarness();
    let state = awaitingSession();
    let protocolErrorCount = 0;
    const dispatch = (event: RunEvent) => { state = reduceRunState(state, event); };
    installPublicRuntimeLifecycle(harness.socket, {
      getRunState: () => state,
      getSelectedRegistrySourceId: () => 'source-ut', dispatch,
      onRuntimeResult: () => undefined,
      onProtocolError: () => { protocolErrorCount += 1; },
    });
    harness.trigger('parser_output', { reason_code: 'public_registry_runtime_started', session_id: 'owned-session' });
    harness.trigger('parser_output', {
      reason_code: 'public_registry_checkpoint_updated', session_id: 'owned-session', registry_source_id: 'source-ut',
      checkpoint: { checkpoint_id: 'not-a-checkpoint', sequence: 1, state: 'complete', label: 'Nope', reason_code: null, summary: null, evidence_count: 1, requires_action: false, action_type: null, updated_at: null },
    });
    harness.trigger('parser_output', {
      reason_code: 'public_registry_action_required', session_id: 'owned-session', registry_source_id: 'source-ut',
      action: { prompt_id: '', checkpoint_id: 'source.acquire', action_type: 'challenge', summary: 'Malformed.' },
    });
    expect(protocolErrorCount).toBe(2);
    expect(state.context.checkpoints.every(({ sequence }) => sequence === 0)).toBe(true);
    expect(state.context.actionRequired).toBeNull();
  });

});
