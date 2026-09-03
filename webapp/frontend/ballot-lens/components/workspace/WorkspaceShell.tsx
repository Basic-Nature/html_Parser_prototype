import { useState } from 'react';

import {
  registrySourceLabel,
  type PublicRegistrySource,
} from '../../contracts/registry';
import type { PublicRuntimeResult } from '../../contracts/publicRuntime';
import type { RunState } from '../../contracts/runtime';
import {
  WorkspaceViews,
  type WorkspaceTab,
} from './WorkspaceViews';

const workspaceTabs: readonly WorkspaceTab[] = [
  'Results',
  'Validation',
  'Metadata',
  'Provenance',
] as const;

interface WorkspaceShellProps {
  readonly selectedSource: PublicRegistrySource | null;
  readonly runState: RunState;
  readonly runtimeResult: PublicRuntimeResult | null;
  readonly dataApiUrl: string;
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
    case 'running':
      return 'Parser running';
    case 'disconnected':
      return 'Connection interrupted';
    case 'terminal':
      return 'Terminal result received';
    default:
      return 'No active run';
  }
}

export function WorkspaceShell({
  selectedSource,
  runState,
  runtimeResult,
  dataApiUrl,
  canRun,
  submitError,
  onRun,
}: WorkspaceShellProps) {
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('Results');
  const selectedLabel = selectedSource
    ? registrySourceLabel(selectedSource)
    : runState.context.sourceSummary?.displayLabel ?? null;
  const trustedTerminalReady = runState.status === 'terminal' && runState.context.runMode !== null && runState.context.runMode !== 'public_registry';
  const trustedPersistedOutputs = trustedTerminalReady ? runState.context.outputs.filter((output) => output.persistence === 'persisted') : [];

  return (
    <section className="blf2-workspace" aria-label="Parser workspace">
      <header className="blf2-workspace-header">
        <div>
          <span className="blf2-kicker">Parser workspace</span>
          <h1>{selectedLabel ?? 'Select an approved source to begin'}</h1>
          <p>
            {runState.context.runMode === 'public_registry' ? 'Approved-source submission sends only the selected registry ID. ' : 'Trusted submission uses the existing certificate-gated URL/upload path. '}
            The server owns the session and structured checkpoint evidence.
          </p>
        </div>
        <div className="blf2-workspace-state" aria-label="Workspace state">
          <span>{runStatusLabel(runState)}</span>
          <span>
            {runState.context.sessionId
              ? `Session ${runState.context.sessionId}`
              : 'Awaiting server session'}
          </span>
          <button
            type="button"
            className="blf2-run-action"
            disabled={!canRun}
            onClick={onRun}
          >
            {runState.status === 'terminal'
              ? 'Run complete'
              : runState.status === 'running'
                ? 'Parser running'
                : runState.status === 'awaiting_session'
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
          <span className="blf2-result-state">
            {runtimeResult ? 'Terminal result received' : 'Awaiting parser output'}
          </span>
        </div>

        {runtimeResult ? (
          <div className="blf2-empty-result blf2-result-ready" role="status">
            <span className="blf2-empty-orbit" aria-hidden="true"><span /></span>
            <strong>Safe parser result is ready</strong>
            <p>
              {runtimeResult.outputs.reduce(
                (total, output) => total + output.row_count,
                0,
              )} rows across {runtimeResult.outputs.length} memory-only preview
              outputs. Use the workspace views below to inspect the real result,
              validation signals, safe metadata, and retained provenance.
            </p>
          </div>
        ) : trustedTerminalReady ? (
          <div className="blf2-empty-result blf2-result-ready" role="status"><strong>Trusted parser run is complete</strong><p>{trustedPersistedOutputs.length} persisted output artifact(s) reported by the existing trusted writer. No preview rows or vote totals are synthesized.</p></div>
        ) : (
          <div className="blf2-empty-result" role="status">
            <span className="blf2-empty-orbit" aria-hidden="true"><span /></span>
            <strong>No parser result yet</strong>
            <p>
              Results appear here only after an owned parser run publishes real
              extracted data. The shell does not fabricate preview rows or vote
              totals.
            </p>
          </div>
        )}

        <div className="blf2-data-guardrails" aria-label="Result guardrails">
          <span>NULL preserved</span>
          <span>No precinct inference</span>
          <span>Provenance retained</span>
        </div>
      </section>

      <div className="blf2-workspace-tabs" role="tablist" aria-label="Workspace views">
        {workspaceTabs.map((tab) => (
          <button
            key={tab}
            type="button"
            role="tab"
            aria-selected={tab === activeTab}
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
      </div>

      <WorkspaceViews
        activeTab={activeTab}
        selectedSource={selectedSource}
        runtimeResult={runtimeResult}
        dataApiUrl={dataApiUrl}
      />
    </section>
  );
}
