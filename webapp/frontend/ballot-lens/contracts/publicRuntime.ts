import {
  isRecord,
  parsePublicRegistrySource,
  type PublicRegistrySource,
} from './registry';

export const PUBLIC_MEMORY_PREVIEW_CONTRACT =
  'ballot_lens_public_memory_preview_v1' as const;
export const PUBLIC_RUNTIME_RESULT_CONTRACT =
  'ballot_lens_public_runtime_result_v1' as const;
export const PUBLIC_EXECUTION_CONTEXT_CONTRACT =
  'ballot_lens_public_execution_context_v1' as const;

export const PUBLIC_TERMINAL_REASON_CODES = Object.freeze([
  'public_download_fallback_disabled',
  'public_memory_preview_missing',
  'public_challenge_assist_disabled',
] as const);

export type PublicTerminalReasonCode =
  typeof PUBLIC_TERMINAL_REASON_CODES[number];

export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export type JsonValue =
  | null
  | string
  | number
  | boolean
  | readonly JsonValue[]
  | JsonObject;

export interface PublicRunProgress {
  readonly type: 'run_progress';
  readonly processed: number;
  readonly total_entries: number;
  readonly status_counts: Readonly<Record<string, number>>;
}

export interface PublicMemoryPreview {
  readonly contract: typeof PUBLIC_MEMORY_PREVIEW_CONTRACT;
  readonly registry_source_id: string;
  readonly source: PublicRegistrySource;
  readonly headers: readonly string[];
  readonly rows: readonly Readonly<Record<string, JsonValue>>[];
  readonly row_count: number;
  readonly output_mode: 'MEMORY_PREVIEW_ONLY';
  readonly download_available: false;
  readonly persistent_output: false;
  readonly execution_context_contract:
    typeof PUBLIC_EXECUTION_CONTEXT_CONTRACT;
  readonly progress?: readonly PublicRunProgress[];
}

export interface PublicRuntimeResult {
  readonly contract: typeof PUBLIC_RUNTIME_RESULT_CONTRACT;
  readonly registry_source_id: string;
  readonly source: PublicRegistrySource;
  readonly outputs: readonly PublicMemoryPreview[];
  readonly status_counts: Readonly<Record<string, number>>;
  readonly terminal_status: string | null;
  readonly terminal_reason_code: PublicTerminalReasonCode | null;
  readonly download_available: false;
  readonly persistent_output: false;
}

function hasExactKeys(
  record: Record<string, unknown>,
  allowedKeys: readonly string[],
): boolean {
  const keys = Object.keys(record).sort();
  const allowed = [...allowedKeys].sort();
  return (
    keys.length === allowed.length
    && keys.every((key, index) => key === allowed[index])
  );
}

function isJsonValue(value: unknown, depth = 0): value is JsonValue {
  if (depth > 32) return false;
  if (
    value === null
    || typeof value === 'string'
    || typeof value === 'boolean'
  ) {
    return true;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value);
  }
  if (Array.isArray(value)) {
    return value.every((entry) => isJsonValue(entry, depth + 1));
  }
  if (isRecord(value)) {
    return Object.entries(value).every(
      ([key, nested]) => (
        typeof key === 'string'
        && isJsonValue(nested, depth + 1)
      ),
    );
  }
  return false;
}

function parseStatusCounts(
  value: unknown,
): Readonly<Record<string, number>> {
  if (!isRecord(value)) {
    throw new Error('Public status counts must be an object');
  }
  const parsed: Record<string, number> = {};
  for (const [key, count] of Object.entries(value)) {
    if (
      !key
      || typeof count !== 'number'
      || !Number.isSafeInteger(count)
      || count < 0
    ) {
      throw new Error('Invalid public status count');
    }
    parsed[key] = count;
  }
  return Object.freeze(parsed);
}

function parseProgress(value: unknown): PublicRunProgress {
  if (!isRecord(value) || !hasExactKeys(value, [
    'processed',
    'status_counts',
    'total_entries',
    'type',
  ])) {
    throw new Error('Unsafe public progress event');
  }
  if (value.type !== 'run_progress') {
    throw new Error('Unexpected public progress event type');
  }
  if (
    typeof value.processed !== 'number'
    || !Number.isSafeInteger(value.processed)
    || value.processed < 0
    || typeof value.total_entries !== 'number'
    || !Number.isSafeInteger(value.total_entries)
    || value.total_entries < 0
    || value.processed > value.total_entries
  ) {
    throw new Error('Invalid public progress counts');
  }

  return Object.freeze({
    type: 'run_progress' as const,
    processed: value.processed,
    total_entries: value.total_entries,
    status_counts: parseStatusCounts(value.status_counts),
  });
}

export function parsePublicMemoryPreview(
  value: unknown,
): PublicMemoryPreview {
  if (!isRecord(value)) {
    throw new Error('Public memory preview must be an object');
  }

  const required = [
    'contract',
    'download_available',
    'execution_context_contract',
    'headers',
    'output_mode',
    'persistent_output',
    'registry_source_id',
    'row_count',
    'rows',
    'source',
  ];
  const allowed = value.progress === undefined
    ? required
    : [...required, 'progress'];

  if (!hasExactKeys(value, allowed)) {
    throw new Error('Unsafe public memory preview projection');
  }
  if (value.contract !== PUBLIC_MEMORY_PREVIEW_CONTRACT) {
    throw new Error('Unexpected public memory preview contract');
  }
  if (
    typeof value.registry_source_id !== 'string'
    || !value.registry_source_id.trim()
  ) {
    throw new Error('Public memory preview requires a source id');
  }
  const registrySourceId = value.registry_source_id.trim();
  const source = parsePublicRegistrySource(value.source, 'preview source');
  if (source.registry_source_id !== registrySourceId) {
    throw new Error('Public memory preview source mismatch');
  }
  if (!Array.isArray(value.headers)) {
    throw new Error('Public memory preview headers must be an array');
  }

  const headers = value.headers.map((header) => {
    if (typeof header !== 'string' || !header.trim()) {
      throw new Error('Public memory preview header is invalid');
    }
    return header;
  });
  if (new Set(headers).size !== headers.length) {
    throw new Error('Public memory preview headers must be unique');
  }

  if (!Array.isArray(value.rows)) {
    throw new Error('Public memory preview rows must be an array');
  }
  const headerSet = new Set(headers);
  const rows = value.rows.map((row) => {
    if (!isRecord(row)) {
      throw new Error('Public memory preview row must be an object');
    }
    const parsed: Record<string, JsonValue> = {};
    for (const [key, nested] of Object.entries(row)) {
      if (!headerSet.has(key) || !isJsonValue(nested)) {
        throw new Error('Unsafe public memory preview row value');
      }
      parsed[key] = nested;
    }
    return Object.freeze(parsed);
  });

  if (
    typeof value.row_count !== 'number'
    || !Number.isSafeInteger(value.row_count)
    || value.row_count !== rows.length
  ) {
    throw new Error('Public memory preview row count mismatch');
  }
  if (value.output_mode !== 'MEMORY_PREVIEW_ONLY') {
    throw new Error('Public output mode is not memory-only');
  }
  if (value.download_available !== false || value.persistent_output !== false) {
    throw new Error('Public preview persistence/download policy drift');
  }
  if (value.execution_context_contract !== PUBLIC_EXECUTION_CONTEXT_CONTRACT) {
    throw new Error('Unexpected public execution context contract');
  }

  let progress: readonly PublicRunProgress[] | undefined;
  if (value.progress !== undefined) {
    if (!Array.isArray(value.progress)) {
      throw new Error('Public preview progress must be an array');
    }
    progress = Object.freeze(value.progress.map(parseProgress));
  }

  return Object.freeze({
    contract: PUBLIC_MEMORY_PREVIEW_CONTRACT,
    registry_source_id: registrySourceId,
    source,
    headers: Object.freeze(headers),
    rows: Object.freeze(rows),
    row_count: rows.length,
    output_mode: 'MEMORY_PREVIEW_ONLY' as const,
    download_available: false as const,
    persistent_output: false as const,
    execution_context_contract: PUBLIC_EXECUTION_CONTEXT_CONTRACT,
    ...(progress ? { progress } : {}),
  });
}

export function parsePublicRuntimeResult(
  value: unknown,
): PublicRuntimeResult {
  if (!isRecord(value) || !hasExactKeys(value, [
    'contract',
    'download_available',
    'outputs',
    'persistent_output',
    'registry_source_id',
    'source',
    'status_counts',
    'terminal_reason_code',
    'terminal_status',
  ])) {
    throw new Error('Unsafe public runtime result projection');
  }
  if (value.contract !== PUBLIC_RUNTIME_RESULT_CONTRACT) {
    throw new Error('Unexpected public runtime result contract');
  }
  if (
    typeof value.registry_source_id !== 'string'
    || !value.registry_source_id.trim()
  ) {
    throw new Error('Public runtime result requires a source id');
  }
  const registrySourceId = value.registry_source_id.trim();
  const source = parsePublicRegistrySource(value.source, 'runtime source');
  if (source.registry_source_id !== registrySourceId) {
    throw new Error('Public runtime result source mismatch');
  }
  if (!Array.isArray(value.outputs)) {
    throw new Error('Public runtime outputs must be an array');
  }
  const outputs = Object.freeze(value.outputs.map((output) => {
    const preview = parsePublicMemoryPreview(output);
    if (preview.registry_source_id !== registrySourceId) {
      throw new Error('Public runtime output source mismatch');
    }
    return preview;
  }));

  if (
    value.terminal_status !== null
    && (typeof value.terminal_status !== 'string'
      || !value.terminal_status.trim())
  ) {
    throw new Error('Invalid public terminal status');
  }

  let terminalReason: PublicTerminalReasonCode | null = null;
  if (value.terminal_reason_code !== null) {
    if (
      typeof value.terminal_reason_code !== 'string'
      || !PUBLIC_TERMINAL_REASON_CODES.includes(
        value.terminal_reason_code as PublicTerminalReasonCode,
      )
    ) {
      throw new Error('Unallowlisted public terminal reason');
    }
    terminalReason = value.terminal_reason_code as PublicTerminalReasonCode;
  }

  if (value.download_available !== false || value.persistent_output !== false) {
    throw new Error('Public runtime persistence/download policy drift');
  }

  return Object.freeze({
    contract: PUBLIC_RUNTIME_RESULT_CONTRACT,
    registry_source_id: registrySourceId,
    source,
    outputs,
    status_counts: parseStatusCounts(value.status_counts),
    terminal_status:
      value.terminal_status === null ? null : value.terminal_status.trim(),
    terminal_reason_code: terminalReason,
    download_available: false as const,
    persistent_output: false as const,
  });
}
