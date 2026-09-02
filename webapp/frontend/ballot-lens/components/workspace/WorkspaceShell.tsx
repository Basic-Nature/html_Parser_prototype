import {
  registrySourceLabel,
  type PublicRegistrySource,
} from '../../contracts/registry';
import type { RunState } from '../../contracts/runtime';

const workspaceTabs = [
  'Results',
  'Validation',
  'Metadata',
  'Provenance',
] as const;

interface WorkspaceShellProps {
  readonly selectedSource: PublicRegistrySource | null;
  readonly runState: RunState;
  readonly canRun: boolean;
  readonly submitError: string | null;
  readonly onRun: () => void;
}

function runStatusLabel(runState: RunState): string {
  switch (runState.status) {
    case 'source_selected':
      return 'Source selected';
    case 'submitting':
      return 'Submitting command';
    case 'awaiting_session':
      return 'Awaiting owned session';
    default:
      return 'No active run';
  }
}

export function WorkspaceShell({
  selectedSource,
  runState,
  canRun,
  submitError,
  onRun,
}: WorkspaceShellProps) {
  const selectedLabel = selectedSource
    ? registrySourceLabel(selectedSource)
    : null;

  return (
    <section className="blf2-workspace" aria-label="Parser workspace">
      <header className="blf2-workspace-header">
        <div>
          <span className="blf2-kicker">Parser workspace</span>
          <h1>{selectedLabel ?? 'Select an approved source to begin'}</h1>
          <p>
            Approved-source submission sends only the selected registry ID.
            Session correlation and parser results remain deferred.
          </p>
        </div>
        <div className="blf2-workspace-state" aria-label="Workspace state">
          <span>{runStatusLabel(runState)}</span>
          <span>No owned session</span>
          <button
            type="button"
            className="blf2-run-action"
            disabled={!canRun}
            onClick={onRun}
          >
            {runState.status === 'awaiting_session'
              ? 'Submission accepted'
              : 'Run approved source'}
          </button>
        </div>
      </header>

      {submitError && (
        <p className="blf2-submit-error" role="alert">{submitError}</p>
      )}

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
