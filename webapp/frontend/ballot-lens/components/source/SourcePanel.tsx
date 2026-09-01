interface SourcePanelProps {
  readonly trustedControls: boolean;
}

const sourceModes = [
  ['Registry', 'Approved public sources', false],
  ['Upload', 'Local election artifact', true],
  ['URL Library', 'Approved trusted targets', true],
  ['Worklist', 'Governed queue handoff', true],
] as const;

export function SourcePanel({ trustedControls }: SourcePanelProps) {
  return (
    <aside className="blf2-source" aria-label="Source">
      <div className="blf2-panel-heading">
        <div><span className="blf2-kicker">Input</span><h2>Source</h2></div>
        <span className="blf2-panel-state">Not selected</span>
      </div>

      <nav className="blf2-source-nav" aria-label="Source modes">
        {sourceModes.map(([label, description, trustedOnly], index) => {
          const available = !trustedOnly || trustedControls;
          return (
            <button
              key={label}
              type="button"
              className="blf2-source-mode"
              data-active={index === 0 ? 'true' : 'false'}
              disabled
              aria-current={index === 0 ? 'page' : undefined}
            >
              <span className="blf2-source-mode-copy">
                <strong>{label}</strong>
                <small>{description}</small>
              </span>
              <span className="blf2-source-mode-meta">
                {trustedOnly ? (available ? 'Trusted' : 'Locked') : 'Public'}
              </span>
            </button>
          );
        })}
      </nav>

      <section className="blf2-selection-card" aria-label="Selected source">
        <span className="blf2-kicker">Selected source</span>
        <strong>No source selected</strong>
        <p>
          F2-C is presentation-only. Registry discovery is wired in F2-D,
          after this shell is accepted.
        </p>
        <dl>
          <div><dt>Authority</dt><dd>Registry boundary preserved</dd></div>
          <div><dt>Execution</dt><dd>Not connected</dd></div>
        </dl>
      </section>
    </aside>
  );
}
