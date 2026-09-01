export function DiagnosticsDrawer() {
  return (
    <details className="blf2-diagnostics">
      <summary>
        <span>
          <strong>Diagnostics &amp; audit trail</strong>
          <small>Raw runtime evidence stays separate from the main workspace.</small>
        </span>
        <span className="blf2-diagnostics-meta">
          <span>0 events</span>
          <span>Dormant</span>
        </span>
      </summary>

      <div className="blf2-diagnostics-body">
        <strong>No runtime events in F2-D discovery.</strong>
        <p>
          Structured diagnostics wiring remains deferred to F2-I. This drawer
          reserves the correct visual location without claiming live telemetry.
        </p>
      </div>
    </details>
  );
}
