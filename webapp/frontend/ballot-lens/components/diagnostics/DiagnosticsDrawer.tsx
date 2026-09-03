import type { RunState } from '../../contracts/runtime';
import {
  getSessionHistoryEntry,
  type SessionHistory,
} from '../../state/sessionHistory';
import { SessionSwitcher } from '../sessions/SessionSwitcher';

interface DiagnosticsDrawerProps {
  readonly runState: RunState;
  readonly sessionHistory: SessionHistory;
  readonly selectedSessionId: string | null;
  readonly onSelectSession: (sessionId: string) => void;
}

export function DiagnosticsDrawer({
  runState,
  sessionHistory,
  selectedSessionId,
  onSelectSession,
}: DiagnosticsDrawerProps) {
  const activeSessionId = runState.context.sessionId;
  const selectedEntry = (
    getSessionHistoryEntry(sessionHistory, selectedSessionId)
    ?? getSessionHistoryEntry(sessionHistory, activeSessionId)
  );

  return (
    <details className="blf2-diagnostics">
      <summary>
        <span>
          <strong>Diagnostics &amp; audit trail</strong>
          <small>Raw runtime evidence stays separate from the main workspace.</small>
        </span>
        <span className="blf2-diagnostics-meta">
          <span>{activeSessionId ? 'Owned session' : 'No session'}</span>
          <span>{runState.status}</span>
        </span>
      </summary>

      <div className="blf2-diagnostics-body">
        <SessionSwitcher
          sessions={sessionHistory}
          activeSessionId={activeSessionId}
          selectedSessionId={selectedEntry?.sessionId ?? null}
          onSelect={onSelectSession}
        />

        {!selectedEntry ? (
          <>
            <strong>No correlated runtime events yet.</strong>
            <p>
              Select and submit an execution-authorized source to begin the
              server-owned session lifecycle.
            </p>
          </>
        ) : (
          <section
            className="blf2-session-detail"
            aria-label="Selected session diagnostics"
          >
            <div className="blf2-session-detail-head">
              <div>
                <span className="blf2-kicker">Selected session</span>
                <strong>{selectedEntry.sessionId}</strong>
              </div>
              <span>
                {selectedEntry.sessionId === activeSessionId
                  ? 'Active authority'
                  : 'Historical view'}
              </span>
            </div>

            <p>
              Server-created session correlated. Connection:{' '}
              {selectedEntry.connectionState}. This history is an app-observed,
              view-only diagnostic record. Selecting an older session never
              changes parser authority or the active server-owned session.
            </p>

            <dl className="blf2-session-facts">
              <div>
                <dt>Mode</dt>
                <dd>{selectedEntry.runMode ?? '—'}</dd>
              </div>
              <div>
                <dt>Source</dt>
                <dd>{selectedEntry.sourceLabel ?? '—'}</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd>{selectedEntry.status}</dd>
              </div>
              <div>
                <dt>Checkpoints</dt>
                <dd>{selectedEntry.checkpointCount}</dd>
              </div>
              <div>
                <dt>Outputs</dt>
                <dd>{selectedEntry.outputCount}</dd>
              </div>
            </dl>

            <ol className="blf2-session-events" aria-label="Observed session events">
              {selectedEntry.observations.map((observation) => (
                <li key={observation.index}>
                  <span>#{observation.index}</span>
                  <strong>{observation.eventType}</strong>
                  <small>
                    {observation.checkpointId
                      ? `${observation.checkpointId} · sequence ${observation.checkpointSequence ?? '—'}`
                      : observation.status}
                  </small>
                </li>
              ))}
            </ol>
          </section>
        )}
      </div>
    </details>
  );
}
