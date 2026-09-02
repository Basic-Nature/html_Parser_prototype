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
          <span>{runState.context.sessionId ? 'Owned session' : 'No session'}</span>
          <span>{runState.status}</span>
        </span>
      </summary>

      <div className="blf2-diagnostics-body">
        <strong>
          {runState.context.sessionId
            ? 'Server-created session correlated.'
            : 'No correlated runtime events yet.'}
        </strong>
        {runState.context.sessionId ? (
          <p>
            Connection: {runState.context.connectionState}. Terminal metadata
            and safe output summaries are app-owned; raw diagnostics remain
            deferred to the session workspace.
          </p>
        ) : (
          <p>
            Select and submit an execution-authorized source to begin the
            server-owned session lifecycle.
          </p>
        )}
      </div>
    </details>
  );
}
