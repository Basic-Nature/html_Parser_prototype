import { describe, expect, it } from 'vitest';
import { createActor } from 'xstate';

import {
  CHECKPOINT_DEFINITIONS,
  type ParserCheckpoint,
} from '../contracts/checkpoints';
import type {
  RunEvent,
  SourceSummary,
} from '../contracts/runtime';
import {
  createInitialRunState,
  reduceRunState,
  runActorLogic,
} from '../state/runMachine';
import {
  canSubmit,
  hasUnresolvedAction,
  isTerminal,
  ownsActiveSession,
  selectCurrentCheckpoint,
} from '../state/selectors';

const publicSource: SourceSummary = {
  runMode: 'public_registry',
  displayLabel: 'Utah 2024 President',
  registrySourceId:
    'blsrc_v1_267d575a499b910deacc0c4ee660e696aeeab9c162126fe17625239be15f615c',
};

function toAwaitingSession() {
  let state = createInitialRunState('connected');
  state = reduceRunState(state, {
    type: 'SOURCE_SELECTED',
    runMode: 'public_registry',
    sourceSummary: publicSource,
  });
  state = reduceRunState(state, { type: 'SUBMIT_REQUESTED' });
  state = reduceRunState(state, { type: 'SUBMISSION_ACCEPTED' });
  return state;
}

function toRunning(sessionId = 'owned-session') {
  return reduceRunState(toAwaitingSession(), {
    type: 'SESSION_CORRELATED',
    runMode: 'public_registry',
    sessionId,
    startedAt: '2026-09-01T21:00:00Z',
  });
}

function checkpoint(
  sequence: number,
  state: ParserCheckpoint['state'] = 'active',
): ParserCheckpoint {
  return {
    checkpointId: 'source.resolve',
    sequence,
    state,
    label: 'Resolve Source',
    reasonCode: null,
    summary: 'Source registry authority resolved.',
    evidenceCount: 1,
    requiresAction: false,
    actionType: null,
    updatedAt: '2026-09-01T21:00:01Z',
  };
}

describe('Ballot Lens F2-B run-state authority', () => {
  it('starts idle with the canonical checkpoint order', () => {
    const state = createInitialRunState();
    expect(state.status).toBe('idle');
    expect(state.context.sessionId).toBeNull();
    expect(state.context.checkpoints.map(({ checkpointId }) => checkpointId))
      .toEqual(CHECKPOINT_DEFINITIONS.map(({ id }) => id));
    expect(state.context.checkpoints.every(({ sequence }) => sequence === 0))
      .toBe(true);
  });

  it('requires an opaque public registry source id', () => {
    const initial = createInitialRunState();
    const invalidSource: SourceSummary = {
      runMode: 'public_registry',
      displayLabel: 'Invalid public source',
    };

    const next = reduceRunState(initial, {
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: invalidSource,
    });
    expect(next).toBe(initial);
  });

  it('moves through source selection, submit, and awaiting-session states', () => {
    let state = createInitialRunState('connected');

    state = reduceRunState(state, {
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: publicSource,
    });
    expect(state.status).toBe('source_selected');
    expect(canSubmit(state)).toBe(true);

    state = reduceRunState(state, { type: 'SUBMIT_REQUESTED' });
    expect(state.status).toBe('submitting');

    state = reduceRunState(state, { type: 'SUBMISSION_ACCEPTED' });
    expect(state.status).toBe('awaiting_session');
  });

  it('does not let a mismatched run mode claim the active session', () => {
    const awaiting = toAwaitingSession();
    const foreignClaim = reduceRunState(awaiting, {
      type: 'SESSION_CORRELATED',
      runMode: 'trusted_url',
      sessionId: 'foreign-session',
      startedAt: '2026-09-01T21:00:00Z',
    });

    expect(foreignClaim).toBe(awaiting);
    expect(foreignClaim.context.sessionId).toBeNull();
  });

  it('claims only a correlated session and rejects foreign session events', () => {
    const running = toRunning();
    expect(running.status).toBe('running');
    expect(ownsActiveSession(running, 'owned-session')).toBe(true);

    const foreignCheckpoint = reduceRunState(running, {
      type: 'CHECKPOINT_UPDATED',
      sessionId: 'foreign-session',
      checkpoint: checkpoint(1),
    });

    expect(foreignCheckpoint).toBe(running);
    expect(selectCurrentCheckpoint(foreignCheckpoint)).toBeNull();
  });

  it('accepts only increasing checkpoint sequence numbers', () => {
    let state = toRunning();

    state = reduceRunState(state, {
      type: 'CHECKPOINT_UPDATED',
      sessionId: 'owned-session',
      checkpoint: checkpoint(2, 'complete'),
    });
    expect(selectCurrentCheckpoint(state)?.sequence).toBe(2);

    const stale = reduceRunState(state, {
      type: 'CHECKPOINT_UPDATED',
      sessionId: 'owned-session',
      checkpoint: checkpoint(1, 'warning'),
    });
    expect(stale).toBe(state);
    expect(selectCurrentCheckpoint(stale)?.state).toBe('complete');
  });

  it('keeps an unresolved action until the matching prompt resolves', () => {
    let state = toRunning();

    state = reduceRunState(state, {
      type: 'ACTION_REQUIRED',
      sessionId: 'owned-session',
      action: {
        promptId: 'prompt-1',
        checkpointId: 'contest.select',
        actionType: 'contest_selection',
        summary: 'Choose a contest.',
      },
    });
    expect(hasUnresolvedAction(state)).toBe(true);

    const wrongPrompt = reduceRunState(state, {
      type: 'ACTION_RESOLVED',
      sessionId: 'owned-session',
      promptId: 'prompt-other',
    });
    expect(wrongPrompt).toBe(state);
    expect(hasUnresolvedAction(wrongPrompt)).toBe(true);

    state = reduceRunState(state, {
      type: 'ACTION_RESOLVED',
      sessionId: 'owned-session',
      promptId: 'prompt-1',
    });
    expect(hasUnresolvedAction(state)).toBe(false);
  });

  it('rejects memory-only output that falsely claims download availability', () => {
    const running = toRunning();

    const invalidTerminalEvent = {
      type: 'RUN_TERMINATED',
      sessionId: 'owned-session',
      terminalStatus: 'completed_with_errors',
      terminalReasonCode: 'public_memory_preview_missing',
      statusCounts: { error: 1 },
      outputs: [
        {
          outputId: 'preview',
          label: 'Memory preview',
          persistence: 'memory_only',
          downloadAvailable: true,
        },
      ],
      completedAt: '2026-09-01T21:01:00Z',
    } as unknown as RunEvent;

    const next = reduceRunState(running, invalidTerminalEvent);
    expect(next).toBe(running);
    expect(isTerminal(next)).toBe(false);
  });

  it('accepts an owned terminal event with truthful output persistence', () => {
    const running = toRunning();

    const terminal = reduceRunState(running, {
      type: 'RUN_TERMINATED',
      sessionId: 'owned-session',
      terminalStatus: 'completed_with_errors',
      terminalReasonCode: 'public_memory_preview_missing',
      statusCounts: { error: 1 },
      outputs: [
        {
          outputId: 'preview',
          label: 'Memory preview',
          persistence: 'memory_only',
          downloadAvailable: false,
        },
      ],
      completedAt: '2026-09-01T21:01:00Z',
    });

    expect(terminal.status).toBe('terminal');
    expect(terminal.context.outputs[0]?.persistence).toBe('memory_only');
    expect(terminal.context.outputs[0]?.downloadAvailable).toBe(false);
    expect(isTerminal(terminal)).toBe(true);
  });

  it('preserves active session ownership through disconnect and reconnect', () => {
    let state = toRunning();

    state = reduceRunState(state, { type: 'CONNECTION_LOST' });
    expect(state.status).toBe('disconnected');
    expect(state.resumeStatus).toBe('running');
    expect(state.context.sessionId).toBe('owned-session');

    state = reduceRunState(state, { type: 'CONNECTION_RESTORED' });
    expect(state.status).toBe('running');
    expect(state.resumeStatus).toBeNull();
    expect(state.context.sessionId).toBe('owned-session');
    expect(state.context.connectionState).toBe('connected');
  });

  it('resets terminal state without changing connection authority', () => {
    let state = toRunning();

    state = reduceRunState(state, {
      type: 'RUN_TERMINATED',
      sessionId: 'owned-session',
      terminalStatus: 'success',
      terminalReasonCode: null,
      statusCounts: { success: 1 },
      outputs: [],
      completedAt: '2026-09-01T21:01:00Z',
    });
    expect(state.status).toBe('terminal');

    state = reduceRunState(state, { type: 'RESET' });
    expect(state.status).toBe('idle');
    expect(state.context.sessionId).toBeNull();
    expect(state.context.connectionState).toBe('connected');
  });

  it('runs through the XState actor wrapper using the same reducer authority', () => {
    const actor = createActor(runActorLogic);
    actor.start();

    actor.send({
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: publicSource,
    });
    actor.send({ type: 'SUBMIT_REQUESTED' });
    actor.send({ type: 'SUBMISSION_ACCEPTED' });
    actor.send({
      type: 'SESSION_CORRELATED',
      runMode: 'public_registry',
      sessionId: 'actor-session',
      startedAt: '2026-09-01T21:00:00Z',
    });

    expect(actor.getSnapshot().context.status).toBe('running');
    expect(actor.getSnapshot().context.context.sessionId)
      .toBe('actor-session');

    actor.stop();
  });
});
