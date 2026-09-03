import { describe, expect, it } from 'vitest';

import {
  EMPTY_SESSION_HISTORY,
  captureOwnedSession,
  getSessionHistoryEntry,
} from '../state/sessionHistory';
import {
  createInitialRunState,
  reduceRunState,
} from '../state/runMachine';
import type { RunState } from '../contracts/runtime';

function createOwnedSession(sessionId: string): RunState {
  let state = createInitialRunState();
  state = reduceRunState(state, {
    type: 'SOURCE_SELECTED',
    runMode: 'public_registry',
    sourceSummary: {
      runMode: 'public_registry',
      displayLabel: 'Approved fixture',
      registrySourceId: 'registry-1',
    },
  });
  state = reduceRunState(state, { type: 'SUBMIT_REQUESTED' });
  state = reduceRunState(state, { type: 'SUBMISSION_ACCEPTED' });
  return reduceRunState(state, {
    type: 'SESSION_CORRELATED',
    runMode: 'public_registry',
    sessionId,
    startedAt: '2026-09-03T00:00:00Z',
  });
}

describe('F2-I session history', () => {
  it('captures only a server-correlated owned session', () => {
    const state = createOwnedSession('session-a');
    const history = captureOwnedSession(
      EMPTY_SESSION_HISTORY,
      state,
      {
        type: 'SESSION_CORRELATED',
        runMode: 'public_registry',
        sessionId: 'session-a',
        startedAt: '2026-09-03T00:00:00Z',
      },
    );

    expect(history).toHaveLength(1);
    expect(history[0]?.sessionId).toBe('session-a');
    expect(history[0]?.status).toBe('running');
    expect(history[0]?.observations).toHaveLength(1);
    expect(Object.isFrozen(history)).toBe(true);
    expect(Object.isFrozen(history[0])).toBe(true);
  });

  it('retains separate immutable historical sessions for diagnostic selection', () => {
    const first = createOwnedSession('session-a');
    const withFirst = captureOwnedSession(
      EMPTY_SESSION_HISTORY,
      first,
      {
        type: 'SESSION_CORRELATED',
        runMode: 'public_registry',
        sessionId: 'session-a',
        startedAt: '2026-09-03T00:00:00Z',
      },
    );
    const second = createOwnedSession('session-b');
    const withSecond = captureOwnedSession(
      withFirst,
      second,
      {
        type: 'SESSION_CORRELATED',
        runMode: 'public_registry',
        sessionId: 'session-b',
        startedAt: '2026-09-03T00:00:01Z',
      },
    );

    expect(withSecond.map(({ sessionId }) => sessionId)).toEqual([
      'session-a',
      'session-b',
    ]);
    expect(getSessionHistoryEntry(withSecond, 'session-a')?.sessionId)
      .toBe('session-a');
    expect(getSessionHistoryEntry(withSecond, 'session-b')?.sessionId)
      .toBe('session-b');
  });

  it('does not invent history before the server owns a session', () => {
    const state = createInitialRunState();
    const history = captureOwnedSession(
      EMPTY_SESSION_HISTORY,
      state,
      { type: 'CONNECTION_ESTABLISHED' },
    );
    expect(history).toBe(EMPTY_SESSION_HISTORY);
  });
});
