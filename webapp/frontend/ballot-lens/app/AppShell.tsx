import type { BallotLensBootstrap } from '../contracts/bootstrap';
import { HeaderBar } from '../components/common/HeaderBar';
import { CheckpointRail } from '../components/checkpoints/CheckpointRail';
import { DiagnosticsDrawer } from '../components/diagnostics/DiagnosticsDrawer';
import { SourcePanel } from '../components/source/SourcePanel';
import { WorkspaceShell } from '../components/workspace/WorkspaceShell';

interface AppShellProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function AppShell({ bootstrap }: AppShellProps) {
  return (
    <div className="blf2-app" data-phase={bootstrap.phase}>
      <HeaderBar bootstrap={bootstrap} />
      <main className="blf2-shell">
        <SourcePanel
          trustedControls={bootstrap.trustedControls}
          publicRegistryApi={bootstrap.publicRegistryApi}
        />
        <WorkspaceShell />
        <CheckpointRail />
      </main>
      <DiagnosticsDrawer />
    </div>
  );
}
