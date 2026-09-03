import { CHECKPOINT_DEFINITIONS, isCheckpointId, type CheckpointState, type ParserCheckpoint } from '../contracts/checkpoints';
import { isRecord } from '../contracts/registry';
import { parsePublicRuntimeResult, type PublicRuntimeResult } from '../contracts/publicRuntime';
import type { ActionRequired } from '../contracts/runtime';

export const PUBLIC_SOCKET_OBSERVATION_EVENTS = Object.freeze(['connect', 'disconnect', 'parser_output', 'public_registry_result'] as const);
const CHECKPOINT_STATES = new Set<CheckpointState>(['pending', 'active', 'complete', 'warning', 'error']);
const ACTION_TYPES = new Set<ActionRequired['actionType']>(['contest_selection', 'vote_method_selection', 'challenge', 'other']);
const CHECKPOINT_LABELS = new Map<string, string>(CHECKPOINT_DEFINITIONS.map(({ id, label }) => [id, label]));

export interface PublicRuntimeStartedObservation { readonly kind: 'runtime_started'; readonly sessionId: string; }
export interface PublicRuntimeResultObservation { readonly kind: 'runtime_result'; readonly result: PublicRuntimeResult; }
export interface PublicConnectionObservation { readonly kind: 'connection'; readonly state: 'connected' | 'disconnected'; }
export interface PublicCheckpointObservation { readonly kind: 'checkpoint'; readonly sessionId: string; readonly registrySourceId: string; readonly checkpoint: ParserCheckpoint; }
export interface PublicActionRequiredObservation { readonly kind: 'action_required'; readonly sessionId: string; readonly registrySourceId: string; readonly action: ActionRequired; }
export interface PublicActionResolvedObservation { readonly kind: 'action_resolved'; readonly sessionId: string; readonly registrySourceId: string; readonly promptId: string; }
export type PublicSocketObservation = PublicRuntimeStartedObservation | PublicRuntimeResultObservation | PublicConnectionObservation | PublicCheckpointObservation | PublicActionRequiredObservation | PublicActionResolvedObservation;

function requireNonEmptyString(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`Invalid public runtime ${field}`);
  return value.trim();
}
function optionalString(value: unknown, field: string): string | null {
  if (value === null || value === undefined) return null;
  return requireNonEmptyString(value, field);
}
function requireSessionAndSource(payload: Record<string, unknown>) {
  return Object.freeze({
    sessionId: requireNonEmptyString(payload.session_id, 'session_id'),
    registrySourceId: requireNonEmptyString(payload.registry_source_id, 'registry_source_id'),
  });
}
function parseCheckpoint(payload: Record<string, unknown>): ParserCheckpoint {
  const value = payload.checkpoint;
  if (!isRecord(value)) throw new Error('Invalid public runtime checkpoint payload');
  const checkpointId = requireNonEmptyString(value.checkpoint_id, 'checkpoint_id');
  if (!isCheckpointId(checkpointId)) throw new Error('Unknown public runtime checkpoint');
  const sequence = value.sequence;
  if (typeof sequence !== 'number' || !Number.isInteger(sequence) || sequence <= 0) throw new Error('Invalid public runtime checkpoint sequence');
  const state = value.state;
  if (typeof state !== 'string' || !CHECKPOINT_STATES.has(state as CheckpointState)) throw new Error('Invalid public runtime checkpoint state');
  const label = requireNonEmptyString(value.label, 'checkpoint label');
  if (CHECKPOINT_LABELS.get(checkpointId) !== label) throw new Error('Public runtime checkpoint label mismatch');
  const evidenceCount = value.evidence_count;
  if (typeof evidenceCount !== 'number' || !Number.isInteger(evidenceCount) || evidenceCount < 0) throw new Error('Invalid public runtime checkpoint evidence count');
  if (typeof value.requires_action !== 'boolean') throw new Error('Invalid public runtime requires_action');
  let actionType: string | null = null;
  if (value.action_type !== null && value.action_type !== undefined) {
    actionType = requireNonEmptyString(value.action_type, 'action_type');
    if (!ACTION_TYPES.has(actionType as ActionRequired['actionType'])) throw new Error('Unknown public runtime checkpoint action type');
  }
  if ((value.requires_action && actionType === null) || (!value.requires_action && actionType !== null)) throw new Error('Public runtime checkpoint action fields disagree');
  return Object.freeze({
    checkpointId, sequence, state: state as CheckpointState, label,
    reasonCode: optionalString(value.reason_code, 'reason_code'),
    summary: optionalString(value.summary, 'summary'), evidenceCount,
    requiresAction: value.requires_action, actionType,
    updatedAt: optionalString(value.updated_at, 'updated_at'),
  });
}
function parseActionRequired(payload: Record<string, unknown>): ActionRequired {
  const value = payload.action;
  if (!isRecord(value)) throw new Error('Invalid public runtime action payload');
  const checkpointId = requireNonEmptyString(value.checkpoint_id, 'action checkpoint_id');
  if (!isCheckpointId(checkpointId)) throw new Error('Unknown public runtime action checkpoint');
  const actionType = requireNonEmptyString(value.action_type, 'action_type');
  if (!ACTION_TYPES.has(actionType as ActionRequired['actionType'])) throw new Error('Unknown public runtime action type');
  return Object.freeze({
    promptId: requireNonEmptyString(value.prompt_id, 'prompt_id'), checkpointId,
    actionType: actionType as ActionRequired['actionType'],
    summary: requireNonEmptyString(value.summary, 'action summary'),
  });
}

export function normalizePublicSocketObservation(eventName: string, payload?: unknown): PublicSocketObservation | null {
  if (eventName === 'connect') return Object.freeze({ kind: 'connection' as const, state: 'connected' as const });
  if (eventName === 'disconnect') return Object.freeze({ kind: 'connection' as const, state: 'disconnected' as const });
  if (eventName === 'parser_output') {
    if (!isRecord(payload)) return null;
    if (payload.reason_code === 'public_registry_runtime_started') {
      return Object.freeze({ kind: 'runtime_started' as const, sessionId: requireNonEmptyString(payload.session_id, 'session_id') });
    }
    if (payload.reason_code === 'public_registry_checkpoint_updated') {
      return Object.freeze({ kind: 'checkpoint' as const, ...requireSessionAndSource(payload), checkpoint: parseCheckpoint(payload) });
    }
    if (payload.reason_code === 'public_registry_action_required') {
      return Object.freeze({ kind: 'action_required' as const, ...requireSessionAndSource(payload), action: parseActionRequired(payload) });
    }
    if (payload.reason_code === 'public_registry_action_resolved') {
      return Object.freeze({ kind: 'action_resolved' as const, ...requireSessionAndSource(payload), promptId: requireNonEmptyString(payload.prompt_id, 'prompt_id') });
    }
    return null;
  }
  if (eventName === 'public_registry_result') return Object.freeze({ kind: 'runtime_result' as const, result: parsePublicRuntimeResult(payload) });
  return null;
}
