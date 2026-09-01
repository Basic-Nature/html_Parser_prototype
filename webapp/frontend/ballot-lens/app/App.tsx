import type { BallotLensBootstrap } from '../contracts/bootstrap';
import { AppShell } from './AppShell';

interface AppProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function App({ bootstrap }: AppProps) {
  return <AppShell bootstrap={bootstrap} />;
}
