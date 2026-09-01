import type { BallotLensBootstrap } from '../contracts/bootstrap';

const checkpoints = [
  ['source.resolve', 'Resolve Source'],
  ['provider.detect', 'Provider Detection'],
  ['source.acquire', 'Acquire'],
  ['structure.detect', 'Detect Structure'],
  ['contest.select', 'Contest Selection'],
  ['vote_methods.detect', 'Vote Method Selection'],
  ['normalize.rows', 'Normalize'],
  ['validate.results', 'Validate'],
  ['preview.publish', 'Preview'],
] as const;

interface AppProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function App({ bootstrap }: AppProps) {
  return (
    <div className="blf2-app" data-phase={bootstrap.phase}>
      <header className="blf2-header">
        <div className="blf2-brand">
          <strong>Ballot Lens</strong>
          <span className="blf2-foundation-badge">F2 foundation</span>
        </div>
        <div className="blf2-header-status" aria-label="Ballot Lens status">
          <span className="blf2-mode">{bootstrap.mode.toUpperCase()}</span>
          <span>Realtime: F2-B</span>
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
              Source discovery is intentionally not wired until the F2-D parity
              tranche.
            </p>
          </div>
        </aside>

        <section className="blf2-workspace" aria-label="Parser workspace">
          <div className="blf2-workspace-header">
            <div>
              <span className="blf2-eyebrow">Parser workspace</span>
              <h1>Frontend foundation loaded</h1>
            </div>
            <span className="blf2-safe-state">Legacy remains default</span>
          </div>

          <div className="blf2-result-placeholder" role="status">
            <strong>Live Result / Table</strong>
            <p>
              F2-A proves the build, mount, capability bootstrap, and application
              shell only. It does not execute or observe parser runs.
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
            {checkpoints.map(([id, label]) => (
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
          Diagnostics wiring is deferred to F2-I. F2-A intentionally has no
          Socket.IO listeners, parser commands, or runtime event ownership.
        </p>
      </details>
    </div>
  );
}
