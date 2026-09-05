import type { PublicRegistrySource } from '../contracts/registry';
import type {
  PublicRuntimeResult,
} from '../contracts/publicRuntime';

const CANONICAL_CONTRACT = 'canonical_results_v1' as const;
const CANONICAL_AUTHORITY = 'canonical_production' as const;
const COMPARISON_FIELDS = [
  'jurisdiction_name',
  'jurisdiction_type',
  'precinct',
  'candidate',
  'total_votes',
] as const;

export type CanonicalComparisonOutcome = 'EXACT_MATCH' | 'UNRESOLVED';

export interface CanonicalEnvelope {
  readonly contract: typeof CANONICAL_CONTRACT;
  readonly authority: typeof CANONICAL_AUTHORITY;
  readonly data_source: 'canonical';
  readonly count: number;
  readonly items: readonly Readonly<Record<string, unknown>>[];
}

export interface CanonicalComparison {
  readonly outcome: CanonicalComparisonOutcome;
  readonly reason: string;
  readonly canonical_count: number;
  readonly parser_row_count: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function unresolved(
  reason: string,
  canonicalCount: number,
  parserRowCount: number,
): CanonicalComparison {
  return Object.freeze({
    outcome: 'UNRESOLVED' as const,
    reason,
    canonical_count: canonicalCount,
    parser_row_count: parserRowCount,
  });
}

function parserRowCount(result: PublicRuntimeResult): number {
  return result.outputs.reduce((total, output) => total + output.row_count, 0);
}

export function buildCanonicalReadPath(
  dataApiUrl: string,
  source: PublicRegistrySource,
): string {
  const raw = dataApiUrl.trim();
  if (!raw.startsWith('/') || raw.startsWith('//') || raw.includes('\\')) {
    throw new Error('Canonical API must be a same-origin relative path');
  }

  const base = 'https://electionpulse.invalid';
  const url = new URL(raw, base);
  if (url.origin !== base || url.username || url.password || url.hash) {
    throw new Error('Canonical API escaped the same-origin read boundary');
  }
  if (url.pathname !== '/api/ballotlens-database') {
    throw new Error('Canonical API path is not the approved production read surface');
  }

  if (source.state.trim()) url.searchParams.set('state', source.state.trim());
  if (/^\d{4}$/.test(source.year.trim())) {
    url.searchParams.set('year', source.year.trim());
  }
  if (source.contest.trim()) {
    url.searchParams.set('contest', source.contest.trim());
  }
  url.searchParams.set('limit', '1000');

  return `${url.pathname}${url.search}`;
}

export function parseCanonicalEnvelope(payload: unknown): CanonicalEnvelope {
  if (!isRecord(payload)) {
    throw new Error('Canonical response must be an object');
  }
  if (
    payload.contract !== CANONICAL_CONTRACT
    || payload.authority !== CANONICAL_AUTHORITY
    || payload.data_source !== 'canonical'
  ) {
    throw new Error('Canonical response authority mismatch');
  }
  if (!Array.isArray(payload.items)) {
    throw new Error('Canonical response items must be an array');
  }
  if (
    typeof payload.count !== 'number'
    || !Number.isSafeInteger(payload.count)
    || payload.count < 0
    || payload.count !== payload.items.length
  ) {
    throw new Error('Canonical response count mismatch');
  }

  const semantic = payload.semantic_contract;
  if (!isRecord(semantic)) {
    throw new Error('Canonical semantic contract is missing');
  }
  if (
    semantic.null !== 'preserved_null'
    || semantic.zero !== 'numeric_zero'
    || semantic.null_reason !== 'not_inferred'
  ) {
    throw new Error('Canonical null/zero semantic contract mismatch');
  }

  const items = payload.items.map((item) => {
    if (!isRecord(item)) {
      throw new Error('Canonical item must be an object');
    }
    return Object.freeze({ ...item });
  });

  return Object.freeze({
    contract: CANONICAL_CONTRACT,
    authority: CANONICAL_AUTHORITY,
    data_source: 'canonical' as const,
    count: payload.count,
    items: Object.freeze(items),
  });
}

function hasOwn(record: Readonly<Record<string, unknown>>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key);
}

function comparableFingerprint(
  record: Readonly<Record<string, unknown>>,
): string | null {
  if (!COMPARISON_FIELDS.every((field) => hasOwn(record, field))) {
    return null;
  }

  const jurisdictionName = record.jurisdiction_name;
  const jurisdictionType = record.jurisdiction_type;
  const precinct = record.precinct;
  const candidate = record.candidate;
  const totalVotes = record.total_votes;

  if (
    !(typeof jurisdictionName === 'string' || jurisdictionName === null)
    || !(typeof jurisdictionType === 'string' || jurisdictionType === null)
    || !(typeof precinct === 'string' || precinct === null)
    || !(typeof candidate === 'string' || candidate === null)
    || !(
      totalVotes === null
      || (typeof totalVotes === 'number' && Number.isFinite(totalVotes))
    )
  ) {
    return null;
  }

  return JSON.stringify([
    jurisdictionName,
    jurisdictionType,
    precinct,
    candidate,
    totalVotes,
  ]);
}

function exactSourceAuthority(
  item: Readonly<Record<string, unknown>>,
  source: PublicRegistrySource,
): boolean {
  const expectedYear = /^\d{4}$/.test(source.year.trim())
    ? Number(source.year.trim())
    : null;
  return (
    item.state === source.state
    && item.contest === source.contest
    && expectedYear !== null
    && (item.year === expectedYear || item.election_year === expectedYear)
  );
}

export function compareRuntimeToCanonical(
  result: PublicRuntimeResult,
  canonical: CanonicalEnvelope,
): CanonicalComparison {
  const rows = result.outputs.flatMap((output) => output.rows);
  const rowCount = rows.length;

  if (canonical.count === 0) {
    return unresolved('canonical_no_rows', 0, rowCount);
  }
  if (canonical.count >= 1000) {
    return unresolved('canonical_result_limit_reached', canonical.count, rowCount);
  }
  if (!canonical.items.every((item) => exactSourceAuthority(item, result.source))) {
    return unresolved('canonical_source_scope_not_exact', canonical.count, rowCount);
  }

  const parserFingerprints = rows.map((row) => (
    comparableFingerprint(row as Readonly<Record<string, unknown>>)
  ));
  const canonicalFingerprints = canonical.items.map(comparableFingerprint);

  if (parserFingerprints.some((value) => value === null)) {
    return unresolved('parser_shape_not_deterministically_comparable', canonical.count, rowCount);
  }
  if (canonicalFingerprints.some((value) => value === null)) {
    return unresolved('canonical_shape_not_deterministically_comparable', canonical.count, rowCount);
  }
  if (parserFingerprints.length !== canonicalFingerprints.length) {
    return unresolved('row_count_not_exact', canonical.count, rowCount);
  }

  const parserSorted = [...parserFingerprints].sort();
  const canonicalSorted = [...canonicalFingerprints].sort();
  if (!parserSorted.every((value, index) => value === canonicalSorted[index])) {
    return unresolved('row_values_not_exact', canonical.count, rowCount);
  }

  return Object.freeze({
    outcome: 'EXACT_MATCH' as const,
    reason: 'exact_deterministic_row_match',
    canonical_count: canonical.count,
    parser_row_count: rowCount,
  });
}

export async function fetchCanonicalComparison(
  dataApiUrl: string,
  result: PublicRuntimeResult,
  fetchImpl: typeof fetch = fetch,
  signal?: AbortSignal,
): Promise<CanonicalComparison> {
  const rowCount = parserRowCount(result);
  if (!dataApiUrl.trim()) {
    return unresolved('canonical_api_not_configured', 0, rowCount);
  }

  let readPath: string;
  try {
    readPath = buildCanonicalReadPath(dataApiUrl, result.source);
  } catch {
    return unresolved('canonical_api_boundary_rejected', 0, rowCount);
  }

  try {
    const response = await fetchImpl(readPath, {
      method: 'GET',
      credentials: 'same-origin',
      headers: { Accept: 'application/json' },
      signal,
    });
    if (!response.ok) {
      return unresolved('canonical_read_failed', 0, rowCount);
    }
    const canonical = parseCanonicalEnvelope(await response.json());
    return compareRuntimeToCanonical(result, canonical);
  } catch {
    return unresolved('canonical_read_failed', 0, rowCount);
  }
}
