import { CHECKPOINT_DEFINITIONS } from '../../contracts/checkpoints';

export function CheckpointRail() {
  return (
    <aside className="blf2-checkpoints" aria-label="Parser checkpoints">
      <div className="blf2-panel-heading">
        <div><span className="blf2-kicker">Run progress</span><h2>Checkpoints</h2></div>
        <span className="blf2-panel-state">0 / 9</span>
      </div>

      <div className="blf2-checkpoint-summary">
        <span className="blf2-status-dot" data-state="pending" aria-hidden="true" />
        <div>
          <strong>Awaiting run</strong>
          <small>Checkpoint state is dormant in F2-C.</small>
        </div>
      </div>

      <ol className="blf2-checkpoint-list">
        {CHECKPOINT_DEFINITIONS.map(({ id, label }, index) => (
          <li key={id} data-state="pending">
            <span className="blf2-checkpoint-index" aria-hidden="true">
              {String(index + 1).padStart(2, '0')}
            </span>
            <span className="blf2-checkpoint-marker" aria-hidden="true" />
            <span className="blf2-checkpoint-copy">
              <strong>{label}</strong>
              <small>{id}</small>
            </span>
          </li>
        ))}
      </ol>
    </aside>
  );
}
