import { PublicRegistryBrowser } from './PublicRegistryBrowser';

interface SourcePanelProps {
  readonly trustedControls: boolean;
  readonly publicRegistryApi: string;
}

const sourceModes = [
  ['Registry', 'Approved public sources', false],
  ['Upload', 'Local election artifact', true],
  ['URL Library', 'Approved trusted targets', true],
  ['Worklist', 'Governed queue handoff', true],
] as const;

export function SourcePanel({
  trustedControls,
  publicRegistryApi,
}: SourcePanelProps) {
  return (
    <aside className="blf2-source" aria-label="Source">
      <div className="blf2-panel-heading">
        <div><span className="blf2-kicker">Input</span><h2>Source</h2></div>
        <span className="blf2-panel-state">Discovery</span>
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

      <PublicRegistryBrowser endpoint={publicRegistryApi} />
    </aside>
  );
}
