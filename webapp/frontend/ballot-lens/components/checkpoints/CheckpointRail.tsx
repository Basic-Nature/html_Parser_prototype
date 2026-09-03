import { CHECKPOINT_DEFINITIONS } from '../../contracts/checkpoints';
import type { CheckpointState, ParserCheckpoint } from '../../contracts/checkpoints';
import type { RunState } from '../../contracts/runtime';

interface CheckpointRailProps { readonly runState: RunState; }

function checkpointById(checkpoints: readonly ParserCheckpoint[]): ReadonlyMap<string, ParserCheckpoint> {
  return new Map(checkpoints.map((checkpoint) => [checkpoint.checkpointId, checkpoint]));
}

function summaryState(runState: RunState, current: ParserCheckpoint | undefined): CheckpointState {
  if (current) return current.state;
  if (runState.status === 'terminal') {
    if (runState.context.terminalStatus === 'failed') return 'error';
    if (runState.context.terminalStatus === 'completed_with_errors') return 'warning';
    return 'complete';
  }
  return 'pending';
}

function summaryTitle(runState: RunState, current: ParserCheckpoint | undefined): string {
  if (runState.context.actionRequired) return 'Action required';
  if (runState.status === 'terminal') {
    if (runState.context.terminalStatus === 'success') return 'Run complete';
    if (runState.context.terminalStatus === 'completed_with_errors') return 'Run completed with warnings';
    if (runState.context.terminalStatus === 'cancelled') return 'Run cancelled';
    return 'Run failed';
  }
  if (current) return current.label;
  if (runState.status === 'running') return 'Waiting for checkpoint evidence';
  return 'Awaiting run';
}

function summaryCopy(runState: RunState, current: ParserCheckpoint | undefined): string {
  if (runState.context.actionRequired) return runState.context.actionRequired.summary;
  if (current?.summary) return current.summary;
  if (current?.reasonCode) return current.reasonCode;
  if (runState.status === 'running') return 'Only structured server checkpoint evidence advances this rail.';
  if (runState.status === 'terminal') return runState.context.terminalReasonCode ?? 'Terminal runtime evidence received.';
  return 'Checkpoint state activates only after an owned public run begins.';
}

export function CheckpointRail({ runState }: CheckpointRailProps) {
  const checkpoints = checkpointById(runState.context.checkpoints);
  const completeCount = runState.context.checkpoints.filter(({ state }) => state === 'complete').length;
  const observedCount = runState.context.checkpoints.filter(({ sequence }) => sequence > 0).length;
  const current = runState.context.currentCheckpoint ? checkpoints.get(runState.context.currentCheckpoint) : undefined;
  const actionRequired = runState.context.actionRequired;

  return (
    <aside className="blf2-checkpoints" aria-label="Parser checkpoints">
      <div className="blf2-panel-heading">
        <div><span className="blf2-kicker">Run progress</span><h2>Checkpoints</h2></div>
        <span className="blf2-panel-state">{completeCount} / {CHECKPOINT_DEFINITIONS.length}</span>
      </div>
      <div className="blf2-checkpoint-summary" aria-live="polite" data-observed-count={observedCount}>
        <span className="blf2-status-dot" data-state={summaryState(runState, current)} aria-hidden="true" />
        <div><strong>{summaryTitle(runState, current)}</strong><small>{summaryCopy(runState, current)}</small></div>
      </div>
      {actionRequired ? (
        <section className="blf2-action-required" aria-label="Action required">
          <div><span className="blf2-kicker">Server evidence</span><strong>Action required</strong></div>
          <p>{actionRequired.summary}</p>
          <dl>
            <div><dt>Checkpoint</dt><dd>{actionRequired.checkpointId}</dd></div>
            <div><dt>Type</dt><dd>{actionRequired.actionType}</dd></div>
          </dl>
          <span className="blf2-panel-state">Display only</span>
        </section>
      ) : null}
      <ol className="blf2-checkpoint-list">
        {CHECKPOINT_DEFINITIONS.map(({ id, label }, index) => {
          const checkpoint = checkpoints.get(id);
          const state = checkpoint?.state ?? 'pending';
          const detail = checkpoint?.summary ?? checkpoint?.reasonCode ?? id;
          return (
            <li key={id} data-state={state} data-sequence={checkpoint?.sequence ?? 0}>
              <span className="blf2-checkpoint-index" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
              <span className="blf2-checkpoint-marker" aria-hidden="true" />
              <span className="blf2-checkpoint-copy"><strong>{label}</strong><small>{detail}</small></span>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
