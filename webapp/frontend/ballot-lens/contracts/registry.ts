export const PUBLIC_REGISTRY_CONTRACT =
  'ballot_lens_public_registry_v1' as const;
export const PUBLIC_REGISTRY_CONTRACT_V2 =
  'ballot_lens_public_registry_v2' as const;
const EXECUTION_ID_PATTERN = /^blsrc_v1_[0-9a-f]{64}$/;

export const PUBLIC_REGISTRY_SOURCE_KEYS = [
  'contest',
  'format',
  'registry_category',
  'registry_source_id',
  'scope',
  'state',
  'year',
] as const;

export const PUBLIC_REGISTRY_ROOT_KEYS = [
  'contract',
  'count',
  'execution_enabled',
  'execution_source_id',
  'sources',
] as const;

export const PUBLIC_REGISTRY_V2_ROOT_KEYS = [
  'contract', 'count', 'execution_enabled', 'execution_source_ids', 'sources',
] as const;

export interface PublicRegistrySource {
  readonly contest: string;
  readonly format: string;
  readonly registry_category: 'curated';
  readonly registry_source_id: string;
  readonly scope: string;
  readonly state: string;
  readonly year: string;
}

export interface PublicRegistryEnvelope {
  readonly contract: typeof PUBLIC_REGISTRY_CONTRACT | typeof PUBLIC_REGISTRY_CONTRACT_V2;
  readonly count: number;
  readonly execution_enabled: boolean;
  readonly execution_source_id: string | null;
  readonly execution_source_ids?: readonly string[];
  readonly sources: readonly PublicRegistrySource[];
}

export interface RegistryFilters {
  readonly query: string;
  readonly state: string;
  readonly year: string;
  readonly contest: string;
  readonly scope: string;
  readonly format: string;
}

export type RegistryFacetKey =
  | 'state'
  | 'year'
  | 'contest'
  | 'scope'
  | 'format';

export interface RegistryFacetOption {
  readonly value: string;
  readonly count: number;
  readonly available: boolean;
}

export const EMPTY_REGISTRY_FILTERS: RegistryFilters = Object.freeze({
  query: '',
  state: '',
  year: '',
  contest: '',
  scope: '',
  format: '',
});

function normalizeText(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value).trim();
  }
  return '';
}

function normalizeNullableText(value: unknown): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'string') return null;
  const normalized = value.trim();
  return normalized || null;
}

export function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hasExactKeys(
  source: Record<string, unknown>,
  allowedKeys: readonly string[],
): boolean {
  const keys = Object.keys(source).sort();
  const allowed = [...allowedKeys].sort();
  return (
    keys.length === allowed.length
    && keys.every((key, index) => key === allowed[index])
  );
}

export function hasExactSafeSourceKeys(
  source: Record<string, unknown>,
): boolean {
  return hasExactKeys(source, PUBLIC_REGISTRY_SOURCE_KEYS);
}

export function parsePublicRegistrySource(
  entry: unknown,
  indexLabel = 'source',
): PublicRegistrySource {
  if (!isRecord(entry) || !hasExactSafeSourceKeys(entry)) {
    throw new Error(`Unsafe public registry source projection at ${indexLabel}`);
  }

  const registrySourceId = normalizeText(entry.registry_source_id);
  if (!registrySourceId) {
    throw new Error(`Missing registry source id at ${indexLabel}`);
  }
  if (entry.registry_category !== 'curated') {
    throw new Error(`Non-curated registry source at ${indexLabel}`);
  }

  return Object.freeze({
    contest: normalizeText(entry.contest),
    format: normalizeText(entry.format),
    registry_category: 'curated' as const,
    registry_source_id: registrySourceId,
    scope: normalizeText(entry.scope),
    state: normalizeText(entry.state),
    year: normalizeText(entry.year),
  });
}

export function parsePublicRegistryPayload(
  payload: unknown,
): PublicRegistryEnvelope {
  if (!isRecord(payload)) {
    throw new Error('Public registry payload must be an object');
  }
  const isV2 = payload.contract === PUBLIC_REGISTRY_CONTRACT_V2;
  if (!hasExactKeys(payload, isV2 ? PUBLIC_REGISTRY_V2_ROOT_KEYS : PUBLIC_REGISTRY_ROOT_KEYS)) {
    throw new Error('Unsafe public registry root projection');
  }
  if (!isV2 && payload.contract !== PUBLIC_REGISTRY_CONTRACT) {
    throw new Error('Unexpected public registry contract');
  }
  if (!Array.isArray(payload.sources)) {
    throw new Error('Public registry sources must be an array');
  }

  const sources = Object.freeze(
    payload.sources.map((entry, index) => (
      parsePublicRegistrySource(entry, `index ${index}`)
    )),
  );

  if (
    typeof payload.count !== 'number'
    || !Number.isSafeInteger(payload.count)
    || payload.count < 0
    || payload.count !== sources.length
  ) {
    throw new Error('Public registry count mismatch');
  }
  if (typeof payload.execution_enabled !== 'boolean') {
    throw new Error('Public registry execution flag must be boolean');
  }

  if (isV2) {
    const ids = payload.execution_source_ids;
    if (!Array.isArray(ids) || ids.length > 2 || !ids.every(
      (id) => typeof id === 'string' && EXECUTION_ID_PATTERN.test(id),
    )) {
      throw new Error('Invalid v2 execution source IDs');
    }
    const sourceIds = sources.map((source) => source.registry_source_id);
    if (new Set(sourceIds).size !== sourceIds.length
      || new Set(ids).size !== ids.length
      || ids.some((id) => !sourceIds.includes(id))
      || payload.execution_enabled !== (ids.length > 0)) {
      throw new Error('Invalid v2 execution authority');
    }
    return Object.freeze({
      contract: PUBLIC_REGISTRY_CONTRACT_V2,
      count: payload.count,
      execution_enabled: payload.execution_enabled,
      // Compatibility field is NOT authority for v2 callers.
      execution_source_id: null,
      execution_source_ids: Object.freeze([...ids] as string[]),
      sources,
    });
  }
  const executionSourceId = normalizeNullableText(
    payload.execution_source_id,
  );

  if (payload.execution_enabled) {
    if (!executionSourceId) {
      throw new Error('Enabled public execution requires a source id');
    }
    if (!sources.some(
      (source) => source.registry_source_id === executionSourceId,
    )) {
      throw new Error('Execution source id is absent from safe registry sources');
    }
  } else if (executionSourceId !== null) {
    throw new Error('Disabled public execution cannot project an execution id');
  }

  return Object.freeze({
    contract: PUBLIC_REGISTRY_CONTRACT,
    count: payload.count,
    execution_enabled: payload.execution_enabled,
    execution_source_id: executionSourceId,
    sources,
  });
}

export function registrySourceLabel(source: PublicRegistrySource): string {
  return [
    source.year,
    source.state,
    source.contest,
    source.scope,
    source.format,
  ].filter(Boolean).join(' • ') || 'Approved election source';
}

function matchesQuery(
  source: PublicRegistrySource,
  query: string,
): boolean {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return true;
  return registrySourceLabel(source).toLowerCase().includes(normalized);
}

function matchesFilters(
  source: PublicRegistrySource,
  filters: RegistryFilters,
  ignoredFacet?: RegistryFacetKey,
): boolean {
  if (!matchesQuery(source, filters.query)) return false;

  const facets: readonly RegistryFacetKey[] = [
    'state',
    'year',
    'contest',
    'scope',
    'format',
  ];

  return facets.every((facet) => {
    if (facet === ignoredFacet) return true;
    const selected = filters[facet].trim();
    return !selected || source[facet] === selected;
  });
}

export function filterRegistrySources(
  sources: readonly PublicRegistrySource[],
  filters: RegistryFilters,
): readonly PublicRegistrySource[] {
  return sources.filter((source) => matchesFilters(source, filters));
}

export function getRegistryFacetOptions(
  sources: readonly PublicRegistrySource[],
  filters: RegistryFilters,
  facet: RegistryFacetKey,
): readonly RegistryFacetOption[] {
  const allValues = Array.from(
    new Set(
      sources
        .map((source) => source[facet])
        .filter(Boolean),
    ),
  ).sort((left, right) => left.localeCompare(right, undefined, {
    numeric: true,
    sensitivity: 'base',
  }));

  return allValues.map((value) => {
    const count = sources.filter(
      (source) => (
        source[facet] === value
        && matchesFilters(source, filters, facet)
      ),
    ).length;

    return Object.freeze({
      value,
      count,
      available: count > 0,
    });
  });
}
