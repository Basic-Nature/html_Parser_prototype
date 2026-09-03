export type BallotLensMode = 'public' | 'trusted';
export type SocketTransport = 'websocket' | 'polling';

export interface SocketIoClientBootstrap {
  readonly transports: readonly SocketTransport[];
  readonly upgrade: boolean;
  readonly pingInterval: number;
  readonly pingTimeout: number;
}

export interface BallotLensBootstrap {
  readonly mode: BallotLensMode;
  readonly trustedControls: boolean;
  readonly publicRegistryApi: string;
  readonly dataApiUrl: string;
  readonly uploadedFiles: readonly string[];
  readonly socketIo: SocketIoClientBootstrap;
  readonly phase: 'F2-E4';
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function parsePositiveInteger(value: unknown, field: string): number {
  if (
    typeof value !== 'number'
    || !Number.isSafeInteger(value)
    || value <= 0
  ) {
    throw new Error(`Invalid Socket.IO bootstrap field: ${field}`);
  }
  return value;
}

export function parseSocketIoClientBootstrap(
  raw: string,
): SocketIoClientBootstrap {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error('Invalid Socket.IO bootstrap JSON');
  }
  if (!isRecord(parsed)) {
    throw new Error('Socket.IO bootstrap must be an object');
  }

  const keys = Object.keys(parsed).sort();
  const allowed = [
    'pingInterval',
    'pingTimeout',
    'transports',
    'upgrade',
  ].sort();
  if (
    keys.length !== allowed.length
    || !keys.every((key, index) => key === allowed[index])
  ) {
    throw new Error('Unexpected Socket.IO bootstrap fields');
  }

  if (!Array.isArray(parsed.transports) || parsed.transports.length === 0) {
    throw new Error('Socket.IO transports must be a non-empty array');
  }

  const transports: SocketTransport[] = [];
  for (const entry of parsed.transports) {
    if (entry !== 'websocket' && entry !== 'polling') {
      throw new Error('Unsupported Socket.IO transport');
    }
    if (!transports.includes(entry)) {
      transports.push(entry);
    }
  }

  if (typeof parsed.upgrade !== 'boolean') {
    throw new Error('Socket.IO upgrade flag must be boolean');
  }

  return Object.freeze({
    transports: Object.freeze(transports),
    upgrade: parsed.upgrade,
    pingInterval: parsePositiveInteger(parsed.pingInterval, 'pingInterval'),
    pingTimeout: parsePositiveInteger(parsed.pingTimeout, 'pingTimeout'),
  });
}

function parseUploadedFiles(
  raw: string | undefined,
): readonly string[] {
  if (!raw) {
    return Object.freeze([]);
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new Error('Invalid uploaded-files bootstrap');
  }
  if (!Array.isArray(parsed)) {
    throw new Error('uploaded-files must be array');
  }

  const output: string[] = [];
  for (const entry of parsed) {
    const candidate = typeof entry === 'string'
      ? entry
      : isRecord(entry) && typeof entry.path === 'string'
        ? entry.path
        : isRecord(entry) && typeof entry.name === 'string'
          ? entry.name
          : '';
    const normalized = candidate
      .replace(/\\/g, '/')
      .replace(/^\/+/, '')
      .trim();
    if (
      normalized
      && !normalized.includes('../')
      && !normalized.startsWith('..')
      && !output.includes(normalized)
    ) {
      output.push(normalized);
    }
  }
  return Object.freeze(output);
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
  if (phase !== 'F2-E4') {
    throw new Error('Unexpected Ballot Lens F2 phase');
  }

  return Object.freeze({
    mode: rawMode,
    trustedControls,
    publicRegistryApi: requireDatasetValue(root, 'publicRegistryApi'),
    dataApiUrl: root.dataset.dataApiUrl ?? '',
    uploadedFiles: parseUploadedFiles(root.dataset.uploadedFiles),
    socketIo: parseSocketIoClientBootstrap(
      requireDatasetValue(root, 'socketioConfig'),
    ),
    phase: 'F2-E4',
  });
}
