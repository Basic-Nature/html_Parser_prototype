export const CHECKPOINT_DEFINITIONS = [
  { id: 'source.resolve', label: 'Resolve Source' },
  { id: 'provider.detect', label: 'Provider Detection' },
  { id: 'source.acquire', label: 'Acquire' },
  { id: 'structure.detect', label: 'Detect Structure' },
  { id: 'contest.select', label: 'Contest Selection' },
  { id: 'vote_methods.detect', label: 'Vote Method Selection' },
  { id: 'normalize.rows', label: 'Normalize' },
  { id: 'validate.results', label: 'Validate' },
  { id: 'preview.publish', label: 'Preview' },
] as const;

export type CheckpointId = (typeof CHECKPOINT_DEFINITIONS)[number]['id'];

export type CheckpointState =
  | 'pending'
  | 'active'
  | 'complete'
  | 'warning'
  | 'error';

export interface ParserCheckpoint {
  readonly checkpointId: CheckpointId;
  readonly sequence: number;
  readonly state: CheckpointState;
  readonly label: string;
  readonly reasonCode: string | null;
  readonly summary: string | null;
  readonly evidenceCount: number;
  readonly requiresAction: boolean;
  readonly actionType: string | null;
  readonly updatedAt: string | null;
}

const checkpointIds = new Set<string>(
  CHECKPOINT_DEFINITIONS.map(({ id }) => id),
);

export function isCheckpointId(value: string): value is CheckpointId {
  return checkpointIds.has(value);
}

export function createInitialCheckpoints(): readonly ParserCheckpoint[] {
  return CHECKPOINT_DEFINITIONS.map(({ id, label }) => ({
    checkpointId: id,
    sequence: 0,
    state: 'pending',
    label,
    reasonCode: null,
    summary: null,
    evidenceCount: 0,
    requiresAction: false,
    actionType: null,
    updatedAt: null,
  }));
}
