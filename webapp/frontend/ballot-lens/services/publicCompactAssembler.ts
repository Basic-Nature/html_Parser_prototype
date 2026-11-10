/**
 * P6C5C2 offline candidate: compact contest page reconstruction.
 * NO socket listeners, NO repository integration, NO persistence, NO activation.
 * Wire schema is pinned to public_contest_compact_pager.py and
 * public_contest_page_broker.py accepted by P6C5B2I_REC1B.
 */
export type CompactScalar = null | string | number | boolean;
export type CompactJson = CompactScalar | readonly CompactJson[] | { readonly [key: string]: CompactJson };
export type CompactRow = Readonly<Record<string, CompactJson>>;
export type CompactPhase = 'idle' | 'awaiting_page' | 'awaiting_ack' | 'released' | 'cancelled';
export type CompactCommand =
  | { readonly event: 'public_registry_compact_next'; readonly payload: { readonly cursor: string } }
  | { readonly event: 'public_registry_compact_finish'; readonly payload: { readonly last_cursor: string } }
  | { readonly event: 'public_registry_compact_cancel'; readonly payload: { readonly snapshot_id: string } };

export interface CompactSummary {
  readonly scope: Readonly<Record<'state' | 'year' | 'contest' | 'registry_source_id', string>>;
  readonly sourceSha256: string;
  readonly headers: readonly string[];
  readonly rows: readonly CompactRow[];
  readonly rowCount: number;
  readonly pageCount: number;
  readonly quarantine: Readonly<{ observations: number; votes: number; reason: string | null }>;
  readonly snapshotId: string;
  readonly downloadAvailable: false;
  readonly persistentOutput: false;
}

const MAX_ROWS = 30000;
const MAX_PAGES = 4096;
const MAX_HEADERS = 500;
const MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024;
const MAX_TOTAL_EVENT_BYTES = 8 * 1024 * 1024;
const MAX_FRAME_EVENT_BYTES = 65536 - 1024;
const MAX_PAGE_BUDGET = 49152;
const asciiSha256 = /^[a-f0-9]{64}$/;
const unsafeHeaders = new Set(['__proto__', 'constructor', 'prototype']);

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
function fields(value: unknown, names: readonly string[], what: string): Record<string, unknown> {
  if (!record(value) || Object.keys(value).sort().join('\u0000') !== [...names].sort().join('\u0000')) {
    throw new Error(`Invalid ${what} fields`);
  }
  return value;
}
function positiveInt(v: unknown, max: number, what: string): number {
  if (typeof v !== 'number' || !Number.isSafeInteger(v) || v < 1 || v > max) {
    throw new Error(`Invalid ${what}`);
  }
  return v;
}
function nonnegativeInt(v: unknown, what: string): number {
  if (typeof v !== 'number' || !Number.isSafeInteger(v) || v < 0) {
    throw new Error(`Invalid ${what}`);
  }
  return v;
}
function boundedString(v: unknown, max: number, what: string): string {
  if (typeof v !== 'string' || !v.trim() || v.length > max) throw new Error(`Invalid ${what}`);
  return v;
}
function validJson(v: unknown, depth = 0): v is CompactJson {
  if (depth > 32) return false;
  if (v === null || typeof v === 'string' || typeof v === 'boolean') return true;
  if (typeof v === 'number') return Number.isFinite(v);
  if (Array.isArray(v)) return v.every(x => validJson(x, depth + 1));
  if (record(v)) return Object.entries(v).every(([k, x]) => !unsafeHeaders.has(k) && validJson(x, depth + 1));
  return false;
}
function byteLength(v: unknown): number {
  const encoded = JSON.stringify(v);
  if (encoded === undefined) throw new Error('Unserializable compact event');
  return new TextEncoder().encode(encoded).length;
}

export class PublicCompactAssembler {
  private phaseValue: CompactPhase = 'idle';
  private snapshot = '';
  private expectedPages = 0;
  private expectedRows = 0;
  private receivedPage = 0;
  private receivedRows = 0;
  private encodedBytes = 0;
  private expiresAt = 0;
  private pendingCursor = '';
  private scope: CompactSummary['scope'] | null = null;
  private sourceSha = '';
  private headers: readonly string[] = [];
  private quarantine: CompactSummary['quarantine'] = { observations: 0, votes: 0, reason: null };
  private rowBuffer: CompactRow[] = [];

  get phase(): CompactPhase { return this.phaseValue; }
  get progress(): Readonly<{ pages: number; totalPages: number; rows: number; totalRows: number }> {
    return Object.freeze({ pages: this.receivedPage, totalPages: this.expectedPages, rows: this.receivedRows, totalRows: this.expectedRows });
  }

  begin(raw: unknown, selectedRegistrySourceId: string, nowSeconds: number): CompactCommand {
    if (this.phaseValue !== 'idle') throw new Error('Compact run already begun');
    const event = fields(raw, ['kind', 'manifest', 'first_cursor', 'quarantine', 'download_available', 'persistent_output'], 'manifest event');
    if (event.kind !== 'public_compact_manifest' || event.download_available !== false || event.persistent_output !== false) {
      throw new Error('Unsafe compact manifest event');
    }
    const manifest = fields(event.manifest, ['contract', 'v', 'scope', 'source_sha256', 'headers', 'rows', 'page_count', 'expires_at', 'snapshot_id', 'page_budget_bytes'], 'compact manifest');
    if (manifest.contract !== 'electionpulse_contest_compact_v1' || manifest.v !== 1) throw new Error('Compact contract mismatch');
    const scope = fields(manifest.scope, ['state', 'year', 'contest', 'registry_source_id'], 'scope');
    const sourceId = boundedString(scope.registry_source_id, 200, 'registry source');
    if (sourceId !== selectedRegistrySourceId) throw new Error('Selected source mismatch');
    const scopeTyped = Object.freeze({
      state: boundedString(scope.state, 200, 'state'),
      year: boundedString(scope.year, 200, 'year'),
      contest: boundedString(scope.contest, 200, 'contest'),
      registry_source_id: sourceId,
    });
    if (!Array.isArray(manifest.headers) || !manifest.headers.length || manifest.headers.length > MAX_HEADERS) {
      throw new Error('Invalid compact headers');
    }
    const headers = manifest.headers.map((h: unknown) => boundedString(h, 200, 'header'));
    if (new Set(headers).size !== headers.length || headers.some(h => unsafeHeaders.has(h))) {
      throw new Error('Unsafe or duplicated compact headers');
    }
    const rows = positiveInt(manifest.rows, MAX_ROWS, 'row count');
    const pageCount = positiveInt(manifest.page_count, MAX_PAGES, 'page count');
    if (pageCount > rows) throw new Error('Page count exceeds row count');
    const expires = positiveInt(manifest.expires_at, Number.MAX_SAFE_INTEGER, 'expiry');
    if (!Number.isSafeInteger(nowSeconds) || nowSeconds >= expires) throw new Error('Expired compact manifest');
    const snapshot = boundedString(manifest.snapshot_id, 64, 'snapshot ID');
    const sha = boundedString(manifest.source_sha256, 64, 'source digest');
    if (!asciiSha256.test(sha)) throw new Error('Invalid source digest');
    if (manifest.page_budget_bytes !== MAX_PAGE_BUDGET) throw new Error('Unexpected page budget');
    const q = fields(event.quarantine, ['observations', 'votes', 'reason'], 'quarantine');
    const observations = nonnegativeInt(q.observations, 'quarantine observations');
    const votes = nonnegativeInt(q.votes, 'quarantine votes');
    if (q.reason !== null && (typeof q.reason !== 'string' || q.reason.length > 100)) throw new Error('Invalid quarantine reason');
    if (observations > 0 && !q.reason) throw new Error('Missing quarantine reason');
    const first = boundedString(event.first_cursor, 1200, 'first cursor');
    if (byteLength(event) > MAX_FRAME_EVENT_BYTES) throw new Error('Oversized compact manifest');
    this.phaseValue = 'awaiting_page';
    this.snapshot = snapshot;
    this.expectedRows = rows;
    this.expectedPages = pageCount;
    this.expiresAt = expires;
    this.scope = scopeTyped;
    this.sourceSha = sha;
    this.headers = Object.freeze(headers);
    this.pendingCursor = first;
    this.quarantine = Object.freeze({ observations, votes, reason: q.reason as string | null });
    return Object.freeze({ event: 'public_registry_compact_next', payload: Object.freeze({ cursor: first }) });
  }

  page(raw: unknown, nowSeconds: number): CompactCommand {
    if (this.phaseValue !== 'awaiting_page') throw new Error('Compact page not expected');
    if (!Number.isSafeInteger(nowSeconds) || nowSeconds >= this.expiresAt) throw new Error('Expired compact run');
    const event = fields(raw, ['kind', 'frame'], 'page event');
    if (event.kind !== 'public_compact_page') throw new Error('Wrong compact event');
    const frame = fields(event.frame, ['v', 'i', 'p', 'n', 'd', 'next'], 'page frame');
    if (frame.v !== 1 || frame.i !== this.snapshot || frame.n !== this.expectedPages || frame.p !== this.receivedPage) {
      throw new Error('Missing, duplicated, reordered or foreign page');
    }
    if (!Array.isArray(frame.d) || frame.d.length < 1 || this.receivedRows + frame.d.length > this.expectedRows) {
      throw new Error('Invalid compact page rows');
    }
    if (byteLength(frame) > MAX_PAGE_BUDGET || byteLength(event) > MAX_FRAME_EVENT_BYTES) {
      throw new Error('Compact page exceeds byte budget');
    }
    const encoded = byteLength(event);
    if (this.encodedBytes + encoded > MAX_TOTAL_EVENT_BYTES) throw new Error('Compact event total exceeded');
    const final = this.receivedPage + 1 === this.expectedPages;
    if ((final && frame.next !== null) || (!final && (typeof frame.next !== 'string' || !frame.next || frame.next.length > 1200))) {
      throw new Error('Compact page cursor chain invalid');
    }
    const rows = (frame.d as unknown[]).map((values): CompactRow => {
      if (!Array.isArray(values) || values.length !== this.headers.length || !values.every(v => validJson(v))) {
        throw new Error('Compact page row width/value invalid');
      }
      const row: Record<string, CompactJson> = Object.create(null);
      this.headers.forEach((h, i) => { row[h] = values[i] as CompactJson; });
      return Object.freeze(row);
    });
    const nextRowCount = this.receivedRows + rows.length;
    if (final !== (nextRowCount === this.expectedRows)) throw new Error('Compact page total row mismatch');
    const currentCursor = this.pendingCursor;
    const nextCursor = frame.next;
    this.rowBuffer.push(...rows);
    this.receivedRows = nextRowCount;
    this.receivedPage++;
    this.encodedBytes += encoded;
    if (final) {
      this.phaseValue = 'awaiting_ack';
      return Object.freeze({ event: 'public_registry_compact_finish', payload: Object.freeze({ last_cursor: currentCursor }) });
    }
    this.pendingCursor = nextCursor as string;
    return Object.freeze({ event: 'public_registry_compact_next', payload: Object.freeze({ cursor: this.pendingCursor }) });
  }

  complete(raw: unknown): CompactSummary {
    if (this.phaseValue !== 'awaiting_ack' || !this.scope) throw new Error('Compact completion not expected');
    const event = fields(raw, ['kind', 'released', 'snapshot_id'], 'completion event');
    if (event.kind !== 'public_compact_complete' || event.released !== true || event.snapshot_id !== this.snapshot) {
      throw new Error('Unacknowledged or foreign compact completion');
    }
    if (this.receivedRows !== this.expectedRows || this.receivedPage !== this.expectedPages) {
      throw new Error('Incomplete compact output');
    }
    this.phaseValue = 'released';
    return Object.freeze({
      scope: this.scope, sourceSha256: this.sourceSha, headers: this.headers,
      rows: Object.freeze([...this.rowBuffer]), rowCount: this.receivedRows,
      pageCount: this.receivedPage, quarantine: this.quarantine, snapshotId: this.snapshot,
      downloadAvailable: false as const, persistentOutput: false as const,
    });
  }

  cancelCommand(): CompactCommand {
    if (!['awaiting_page', 'awaiting_ack'].includes(this.phaseValue)) throw new Error('No active compact run');
    return Object.freeze({ event: 'public_registry_compact_cancel', payload: Object.freeze({ snapshot_id: this.snapshot }) });
  }

  cancelled(raw: unknown): void {
    if (!['awaiting_page', 'awaiting_ack'].includes(this.phaseValue)) throw new Error('No active compact run');
    const event = fields(raw, ['kind', 'released'], 'cancellation event');
    if (event.kind !== 'public_compact_cancelled' || event.released !== true) throw new Error('Compact cancellation not confirmed');
    this.phaseValue = 'cancelled';
    this.rowBuffer = [];
  }
}
