import type { RunState } from '../../contracts/runtime';

interface DiagnosticsDrawerProps {
  readonly runState: RunState;
}

export function DiagnosticsDrawer({ runState }: DiagnosticsDrawerProps) {
  return (
    <details className="blf2-diagnostics">
      <summary>
        <span>
          <strong>Diagnostics &amp; audit trail</strong>
          <small>Raw runtime evidence stays separate from the main workspace.</small>
        </span>
        <span className="blf2-diagnostics-meta">
          <span>0 events</span>
          <span>{runState.status}</span>
        </span>
      </summary>

      <div className="blf2-diagnostics-body">
        <strong>No correlated runtime events yet.</strong>
        <p>
          F2-E2 records only app-owned selection and submission state. Session
          correlation, checkpoints, and result handoff remain deferred.
        </p>
      </div>
    </details>
  );
}
