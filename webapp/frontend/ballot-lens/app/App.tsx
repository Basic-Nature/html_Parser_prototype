import type { BallotLensBootstrap } from '../contracts/bootstrap';
import { CHECKPOINT_DEFINITIONS } from '../contracts/checkpoints';

interface AppProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function App({ bootstrap }: AppProps) {
  return (
    <div className="blf2-app" data-phase={bootstrap.phase}>
      <header className="blf2-header">
        <div className="blf2-brand">
          <strong>Ballot Lens</strong>
          <span className="blf2-foundation-badge">F2 contracts</span>
        </div>
        <div className="blf2-header-status" aria-label="Ballot Lens status">
          <span className="blf2-mode">{bootstrap.mode.toUpperCase()}</span>
          <span>Run state: idle</span>
          <span>Session: —</span>
          <a href="/quick_reference">Help</a>
        </div>
      </header>

      <main className="blf2-shell">
        <aside className="blf2-source" aria-label="Source">
          <h2>Source</h2>
          <nav className="blf2-source-nav" aria-label="Source modes">
            <button type="button" aria-current="page">Registry</button>
            <button type="button" disabled={!bootstrap.trustedControls}>
              Upload
            </button>
            <button type="button" disabled={!bootstrap.trustedControls}>
              URL Library
            </button>
            <button type="button" disabled={!bootstrap.trustedControls}>
              Worklist
            </button>
          </nav>
          <div className="blf2-source-context">
            <span>Selected</span>
            <strong>No source selected</strong>
            <p>
              Source discovery remains intentionally unwired until the F2-D
              parity tranche.
            </p>
          </div>
        </aside>

        <section className="blf2-workspace" aria-label="Parser workspace">
          <div className="blf2-workspace-header">
            <div>
              <span className="blf2-eyebrow">Parser workspace</span>
              <h1>Typed run-state contracts ready</h1>
            </div>
            <span className="blf2-safe-state">Legacy remains default</span>
          </div>

          <div className="blf2-result-placeholder" role="status">
            <strong>Live Result / Table</strong>
            <p>
              F2-B establishes normalized runtime contracts, checkpoint order,
              and session-owned run state. Raw Socket.IO events and parser
              commands are still intentionally disconnected from this app.
            </p>
          </div>

          <nav className="blf2-tabs" aria-label="Workspace views">
            <button type="button" aria-current="page">Results</button>
            <button type="button">Validation</button>
            <button type="button">Metadata</button>
            <button type="button">Provenance</button>
          </nav>
        </section>

        <aside className="blf2-checkpoints" aria-label="Parser checkpoints">
          <h2>Checkpoints</h2>
          <ol>
            {CHECKPOINT_DEFINITIONS.map(({ id, label }) => (
              <li key={id}>
                <span className="blf2-checkpoint-dot" aria-hidden="true">○</span>
                <span>
                  <strong>{label}</strong>
                  <small>{id}</small>
                </span>
              </li>
            ))}
          </ol>
        </aside>
      </main>

      <details className="blf2-diagnostics">
        <summary>Diagnostics / raw parser events / audit trail</summary>
        <p>
          Diagnostics wiring remains deferred to F2-I. F2-B defines internal
          normalized event authority only; it still has no Socket.IO listeners,
          parser commands, or runtime event router.
        </p>
      </details>
    </div>
  );
}
