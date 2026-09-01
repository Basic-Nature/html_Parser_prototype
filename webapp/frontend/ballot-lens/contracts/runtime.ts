import type {
  CheckpointId,
  ParserCheckpoint,
} from './checkpoints';

export type RunMode =
  | 'public_registry'
  | 'trusted_url'
  | 'manual_upload'
  | 'worklist';

export type RunStatus =
  | 'idle'
  | 'source_selected'
  | 'submitting'
  | 'awaiting_session'
  | 'running'
  | 'disconnected'
  | 'terminal';

export type ResumableRunStatus =
  | 'submitting'
  | 'awaiting_session'
  | 'running';

export type ConnectionState =
  | 'not_connected'
  | 'connected'
  | 'disconnected';

export type TerminalStatus =
  | 'success'
  | 'completed_with_errors'
  | 'failed'
  | 'cancelled';

export interface SourceSummary {
  readonly runMode: RunMode;
  readonly displayLabel: string;
  readonly registrySourceId?: string;
}

export interface ActionRequired {
  readonly promptId: string;
  readonly checkpointId: CheckpointId;
  readonly actionType:
    | 'contest_selection'
    | 'vote_method_selection'
    | 'challenge'
    | 'other';
  readonly summary: string;
}

export interface MemoryOnlyRunOutput {
  readonly outputId: string;
  readonly label: string;
  readonly persistence: 'memory_only';
  readonly downloadAvailable: false;
}

export interface PersistedRunOutput {
  readonly outputId: string;
  readonly label: string;
  readonly persistence: 'persisted';
  readonly downloadAvailable: boolean;
}

export type RunOutput = MemoryOnlyRunOutput | PersistedRunOutput;

export interface RunContext {
  readonly runMode: RunMode | null;
  readonly sessionId: string | null;
  readonly sourceSummary: SourceSummary | null;
  readonly provider: string | null;
  readonly checkpoints: readonly ParserCheckpoint[];
  readonly currentCheckpoint: CheckpointId | null;
  readonly actionRequired: ActionRequired | null;
  readonly terminalStatus: TerminalStatus | null;
  readonly terminalReasonCode: string | null;
  readonly statusCounts: Readonly<Record<string, number>>;
  readonly outputs: readonly RunOutput[];
  readonly startedAt: string | null;
  readonly completedAt: string | null;
  readonly connectionState: ConnectionState;
}

export interface RunState {
  readonly status: RunStatus;
  readonly resumeStatus: ResumableRunStatus | null;
  readonly context: RunContext;
}

export type RunEvent =
  | {
      readonly type: 'SOURCE_SELECTED';
      readonly runMode: RunMode;
      readonly sourceSummary: SourceSummary;
    }
  | { readonly type: 'SUBMIT_REQUESTED' }
  | { readonly type: 'SUBMISSION_ACCEPTED' }
  | {
      readonly type: 'SESSION_CORRELATED';
      readonly runMode: RunMode;
      readonly sessionId: string;
      readonly startedAt: string;
    }
  | {
      readonly type: 'PROVIDER_IDENTIFIED';
      readonly sessionId: string;
      readonly provider: string;
    }
  | {
      readonly type: 'CHECKPOINT_UPDATED';
      readonly sessionId: string;
      readonly checkpoint: ParserCheckpoint;
    }
  | {
      readonly type: 'ACTION_REQUIRED';
      readonly sessionId: string;
      readonly action: ActionRequired;
    }
  | {
      readonly type: 'ACTION_RESOLVED';
      readonly sessionId: string;
      readonly promptId: string;
    }
  | {
      readonly type: 'RUN_TERMINATED';
      readonly sessionId: string;
      readonly terminalStatus: TerminalStatus;
      readonly terminalReasonCode: string | null;
      readonly statusCounts: Readonly<Record<string, number>>;
      readonly outputs: readonly RunOutput[];
      readonly completedAt: string;
    }
  | { readonly type: 'CONNECTION_ESTABLISHED' }
  | { readonly type: 'CONNECTION_LOST' }
  | { readonly type: 'CONNECTION_RESTORED' }
  | { readonly type: 'RESET' };
