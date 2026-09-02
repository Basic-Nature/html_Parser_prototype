import type { BallotLensBootstrap } from '../../contracts/bootstrap';
import type { RunState } from '../../contracts/runtime';

interface HeaderBarProps {
  readonly bootstrap: BallotLensBootstrap;
  readonly runState: RunState;
}

export function HeaderBar({ bootstrap, runState }: HeaderBarProps) {
  const commandState = runState.status === 'awaiting_session'
    ? 'Command accepted'
    : runState.status === 'submitting'
      ? 'Submitting'
      : 'Submit ready';

  return (
    <header className="blf2-header">
      <div className="blf2-brand-block">
        <span className="blf2-lens-mark" aria-hidden="true"><span /></span>
        <div className="blf2-brand-copy">
          <span>ElectionPulse workspace</span>
          <strong>Ballot Lens</strong>
        </div>
        <span className="blf2-phase-badge">F2-E2 submit</span>
      </div>

      <div className="blf2-header-status" aria-label="Ballot Lens status">
        <span className="blf2-mode-badge">{bootstrap.mode.toUpperCase()}</span>
        <span className="blf2-status-item">
          <span className="blf2-status-dot" data-state="dormant" aria-hidden="true" />
          {commandState}
        </span>
        <span className="blf2-status-item">Approved source only</span>
        <span className="blf2-session">Session <strong>—</strong></span>
        <a className="blf2-help-link" href="/quick_reference">Help</a>
      </div>
    </header>
  );
}
