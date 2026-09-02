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
  normalizePublicSocketObservation,
} from '../services/socketAdapter';

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

  it('constructs transport with autoConnect false and no execution emit API', () => {
    let captured: DormantSocketFactoryOptions | null = null;
    const fakeSocket: DormantBallotLensSocket = {
      connected: false,
      connect() { return this; },
      disconnect() { return this; },
      on() { return this; },
      off() { return this; },
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
    expect('emit' in socket).toBe(false);
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
