import type {
  SessionHistory,
} from '../../state/sessionHistory';

interface SessionSwitcherProps {
  readonly sessions: SessionHistory;
  readonly activeSessionId: string | null;
  readonly selectedSessionId: string | null;
  readonly onSelect: (sessionId: string) => void;
}

export function SessionSwitcher({
  sessions,
  activeSessionId,
  selectedSessionId,
  onSelect,
}: SessionSwitcherProps) {
  return (
    <section
      className="blf2-session-switcher"
      aria-label="Observed parser sessions"
    >
      <div className="blf2-session-switcher-head">
        <strong>Session history</strong>
        <small>View only · current page lifetime</small>
      </div>

      {sessions.length === 0 ? (
        <p>No owned sessions have been observed in this page yet.</p>
      ) : (
        <div className="blf2-session-switcher-list">
          {[...sessions].reverse().map((session) => (
            <button
              key={session.sessionId}
              type="button"
              aria-pressed={session.sessionId === selectedSessionId}
              onClick={() => onSelect(session.sessionId)}
            >
              <strong>{session.sessionId}</strong>
              <small>
                {session.sessionId === activeSessionId
                  ? 'Active'
                  : session.status}
              </small>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
