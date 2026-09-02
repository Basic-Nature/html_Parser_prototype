import {
  isRecord,
} from '../contracts/registry';
import {
  parsePublicRuntimeResult,
  type PublicRuntimeResult,
} from '../contracts/publicRuntime';

export const PUBLIC_SOCKET_OBSERVATION_EVENTS = Object.freeze([
  'connect',
  'disconnect',
  'parser_output',
  'public_registry_result',
] as const);

export interface PublicRuntimeStartedObservation {
  readonly kind: 'runtime_started';
  readonly sessionId: string;
}

export interface PublicRuntimeResultObservation {
  readonly kind: 'runtime_result';
  readonly result: PublicRuntimeResult;
}

export interface PublicConnectionObservation {
  readonly kind: 'connection';
  readonly state: 'connected' | 'disconnected';
}

export type PublicSocketObservation =
  | PublicRuntimeStartedObservation
  | PublicRuntimeResultObservation
  | PublicConnectionObservation;

export function normalizePublicSocketObservation(
  eventName: string,
  payload?: unknown,
): PublicSocketObservation | null {
  if (eventName === 'connect') {
    return Object.freeze({
      kind: 'connection' as const,
      state: 'connected' as const,
    });
  }
  if (eventName === 'disconnect') {
    return Object.freeze({
      kind: 'connection' as const,
      state: 'disconnected' as const,
    });
  }
  if (eventName === 'parser_output') {
    if (
      !isRecord(payload)
      || payload.reason_code !== 'public_registry_runtime_started'
      || typeof payload.session_id !== 'string'
      || !payload.session_id.trim()
    ) {
      return null;
    }
    return Object.freeze({
      kind: 'runtime_started' as const,
      sessionId: payload.session_id.trim(),
    });
  }
  if (eventName === 'public_registry_result') {
    return Object.freeze({
      kind: 'runtime_result' as const,
      result: parsePublicRuntimeResult(payload),
    });
  }
  return null;
}
