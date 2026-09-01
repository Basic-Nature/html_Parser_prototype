export type BallotLensMode = 'public' | 'trusted';

export interface BallotLensBootstrap {
  readonly mode: BallotLensMode;
  readonly trustedControls: boolean;
  readonly publicRegistryApi: string;
  readonly dataApiUrl: string;
  readonly phase: 'F2-C';
}

function requireDatasetValue(
  element: HTMLElement,
  key: keyof DOMStringMap,
): string {
  const value = element.dataset[key];
  if (!value) {
    throw new Error(`Missing Ballot Lens F2 bootstrap field: ${String(key)}`);
  }
  return value;
}

export function readBallotLensBootstrap(
  root: HTMLElement,
): BallotLensBootstrap {
  const rawMode = requireDatasetValue(root, 'mode');
  if (rawMode !== 'public' && rawMode !== 'trusted') {
    throw new Error('Invalid Ballot Lens F2 mode');
  }

  const trustedControls =
    requireDatasetValue(root, 'trustedControls') === '1';

  if ((rawMode === 'trusted') !== trustedControls) {
    throw new Error('Ballot Lens F2 capability bootstrap mismatch');
  }

  const phase = requireDatasetValue(root, 'f2Phase');
  if (phase !== 'F2-C') {
    throw new Error('Unexpected Ballot Lens F2 phase');
  }

  return Object.freeze({
    mode: rawMode,
    trustedControls,
    publicRegistryApi: requireDatasetValue(root, 'publicRegistryApi'),
    dataApiUrl: root.dataset.dataApiUrl ?? '',
    phase: 'F2-C',
  });
}
