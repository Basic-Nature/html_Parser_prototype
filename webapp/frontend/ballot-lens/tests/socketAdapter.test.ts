import { describe, expect, it } from 'vitest';

import {
  parseSocketIoClientBootstrap,
} from '../contracts/bootstrap';
import {
  createDormantBallotLensSocket,
  type DormantBallotLensSocket,
  type DormantSocketFactoryOptions,
} from '../services/socketClient';
import {
  submitApprovedRegistrySource,
} from '../services/publicSubmit';
import {
  normalizePublicSocketObservation,
} from '../services/socketAdapter';
import {
  canSubmitApprovedRegistrySource,
} from '../state/selectors';

const approvedSource = Object.freeze({
  contest: 'General Election',
  format: 'JSON',
  registry_category: 'curated' as const,
  registry_source_id: 'source-ut',
  scope: 'County',
  state: 'UT',
  year: '2024',
});

describe('F2-E1 dormant Socket.IO foundation', () => {
  it('parses only the server-owned safe Socket.IO bootstrap shape', () => {
    const config = parseSocketIoClientBootstrap(JSON.stringify({
      transports: ['websocket', 'polling'],
      upgrade: true,
      pingInterval: 10000,
      pingTimeout: 60000,
    }));
    expect(config.transports).toEqual(['websocket', 'polling']);
    expect(config.upgrade).toBe(true);

    expect(() => parseSocketIoClientBootstrap(JSON.stringify({
      transports: ['websocket'],
      upgrade: true,
      pingInterval: 10000,
      pingTimeout: 60000,
      url: 'https://example.invalid',
    }))).toThrow(/Unexpected Socket.IO bootstrap fields/);
  });

  it('constructs command transport with autoConnect false', () => {
    let captured: DormantSocketFactoryOptions | null = null;
    const emitted: unknown[] = [];
    const fakeSocket: DormantBallotLensSocket = {
      connected: false,
      connect() { return this; },
      disconnect() { return this; },
      on() { return this; },
      off() { return this; },
      emit(event, payload) {
        emitted.push([event, payload]);
        return this;
      },
    };

    const socket = createDormantBallotLensSocket(
      {
        transports: ['websocket', 'polling'],
        upgrade: true,
        pingInterval: 10000,
        pingTimeout: 60000,
      },
      (options) => {
        captured = options;
        return fakeSocket;
      },
    );

    expect(socket).toBe(fakeSocket);
    expect(captured).toEqual({
      autoConnect: false,
      transports: ['websocket', 'polling'],
      upgrade: true,
    });
    expect(emitted).toEqual([]);
  });

  it('submits exactly one approved registry id command', () => {
    let connected = false;
    const emitted: unknown[] = [];
    const fakeSocket: DormantBallotLensSocket = {
      connected: false,
      connect() { connected = true; return this; },
      disconnect() { return this; },
      on() { return this; },
      off() { return this; },
      emit(event, payload) {
        emitted.push([event, payload]);
        return this;
      },
    };

    submitApprovedRegistrySource(fakeSocket, ' source-ut ');

    expect(connected).toBe(true);
    expect(emitted).toEqual([
      ['ballot_lens', { registry_source_id: 'source-ut' }],
    ]);
    expect(() => submitApprovedRegistrySource(fakeSocket, '   '))
      .toThrow(/source id is required/);
  });

  it('requires exact root execution authority for the selected source', () => {
    const envelope = Object.freeze({
      contract: 'ballot_lens_public_registry_v1' as const,
      count: 1,
      execution_enabled: true,
      execution_source_id: 'source-ut',
      sources: Object.freeze([approvedSource]),
    });

    expect(canSubmitApprovedRegistrySource(envelope, approvedSource)).toBe(true);
    expect(canSubmitApprovedRegistrySource({
      ...envelope,
      execution_enabled: false,
      execution_source_id: null,
    }, approvedSource)).toBe(false);
    expect(canSubmitApprovedRegistrySource(envelope, {
      ...approvedSource,
      registry_source_id: 'source-other',
    })).toBe(false);
  });

  it('normalizes server-created public runtime session observation', () => {
    expect(normalizePublicSocketObservation('parser_output', {
      reason_code: 'public_registry_runtime_started',
      session_id: 'session-server-owned',
    })).toEqual({
      kind: 'runtime_started',
      sessionId: 'session-server-owned',
    });
  });

  it('ignores unrelated parser output rather than manufacturing lifecycle', () => {
    expect(normalizePublicSocketObservation('parser_output', {
      reason_code: 'other_status',
      session_id: 'session-x',
    })).toBeNull();
  });

  it('normalizes connection state without connecting anything', () => {
    expect(normalizePublicSocketObservation('connect')).toEqual({
      kind: 'connection',
      state: 'connected',
    });
    expect(normalizePublicSocketObservation('disconnect')).toEqual({
      kind: 'connection',
      state: 'disconnected',
    });
  });
});
