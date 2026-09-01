import type { BallotLensBootstrap } from '../../contracts/bootstrap';

interface HeaderBarProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function HeaderBar({ bootstrap }: HeaderBarProps) {
  return (
    <header className="blf2-header">
      <div className="blf2-brand-block">
        <span className="blf2-lens-mark" aria-hidden="true"><span /></span>
        <div className="blf2-brand-copy">
          <span>ElectionPulse workspace</span>
          <strong>Ballot Lens</strong>
        </div>
        <span className="blf2-phase-badge">F2-D discovery</span>
      </div>

      <div className="blf2-header-status" aria-label="Ballot Lens status">
        <span className="blf2-mode-badge">{bootstrap.mode.toUpperCase()}</span>
        <span className="blf2-status-item">
          <span className="blf2-status-dot" data-state="dormant" aria-hidden="true" />
          Runtime dormant
        </span>
        <span className="blf2-status-item">Discovery active</span>
        <span className="blf2-session">Session <strong>—</strong></span>
        <a className="blf2-help-link" href="/quick_reference">Help</a>
      </div>
    </header>
  );
}
