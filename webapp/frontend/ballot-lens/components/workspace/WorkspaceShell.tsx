const workspaceTabs = [
  'Results',
  'Validation',
  'Metadata',
  'Provenance',
] as const;

export function WorkspaceShell() {
  return (
    <section className="blf2-workspace" aria-label="Parser workspace">
      <header className="blf2-workspace-header">
        <div>
          <span className="blf2-kicker">Parser workspace</span>
          <h1>Select an approved source to begin</h1>
          <p>
            The workspace is ready for F2-D source discovery and F2-E
            execution wiring.
          </p>
        </div>
        <div className="blf2-workspace-state" aria-label="Workspace state">
          <span>No active run</span>
          <span>No owned session</span>
        </div>
      </header>

      <section className="blf2-result-frame" aria-label="Live result workspace">
        <div className="blf2-result-toolbar">
          <div><span className="blf2-kicker">Live result</span><strong>Result table</strong></div>
          <span className="blf2-result-state">Awaiting parser output</span>
        </div>

        <div className="blf2-empty-result" role="status">
          <span className="blf2-empty-orbit" aria-hidden="true"><span /></span>
          <strong>No parser result yet</strong>
          <p>
            Results appear here only after an owned parser run publishes real
            extracted data. F2-C does not fabricate preview rows or vote totals.
          </p>
        </div>

        <div className="blf2-data-guardrails" aria-label="Result guardrails">
          <span>NULL preserved</span>
          <span>No precinct inference</span>
          <span>Provenance retained</span>
        </div>
      </section>

      <div className="blf2-workspace-tabs" role="tablist" aria-label="Workspace views">
        {workspaceTabs.map((tab, index) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={index === 0}
            disabled
          >
            {tab}
          </button>
        ))}
      </div>

      <section className="blf2-tab-panel" role="tabpanel" aria-label="Results">
        <div>
          <span className="blf2-kicker">Results</span>
          <strong>Owned-run output will render here.</strong>
        </div>
        <p>
          Validation, metadata, and provenance stay adjacent to the result
          instead of being scattered across unrelated cards.
        </p>
      </section>
    </section>
  );
}
