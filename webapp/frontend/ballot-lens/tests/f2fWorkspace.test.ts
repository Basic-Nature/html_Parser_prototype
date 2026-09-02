import { describe, expect, it } from 'vitest';

import type { PublicRuntimeResult } from '../contracts/publicRuntime';
import {
  buildCanonicalReadPath,
  compareRuntimeToCanonical,
  fetchCanonicalComparison,
  parseCanonicalEnvelope,
} from '../services/canonicalComparison';
import {
  filterPreviewRows,
  formatResultCell,
} from '../components/workspace/WorkspaceViews';

const source = Object.freeze({
  contest: 'Contract Contest',
  format: 'HTML',
  registry_category: 'curated',
  registry_source_id: 'source-contract',
  scope: 'statewide',
  state: 'ZZ',
  year: '2024',
});

const runtimeResult: PublicRuntimeResult = Object.freeze({
  contract: 'ballot_lens_public_runtime_result_v1',
  registry_source_id: 'source-contract',
  source,
  outputs: Object.freeze([
    Object.freeze({
      contract: 'ballot_lens_public_memory_preview_v1',
      registry_source_id: 'source-contract',
      source,
      headers: Object.freeze([
        'jurisdiction_name',
        'jurisdiction_type',
        'precinct',
        'candidate',
        'total_votes',
      ]),
      rows: Object.freeze([
        Object.freeze({
          jurisdiction_name: 'District 5',
          jurisdiction_type: 'district',
          precinct: null,
          candidate: 'Zero Candidate',
          total_votes: 0,
        }),
      ]),
      row_count: 1,
      output_mode: 'MEMORY_PREVIEW_ONLY',
      download_available: false,
      persistent_output: false,
      execution_context_contract: 'ballot_lens_public_execution_context_v1',
    }),
  ]),
  status_counts: Object.freeze({ success: 1 }),
  terminal_status: 'success',
  terminal_reason_code: null,
  download_available: false,
  persistent_output: false,
});

const canonicalPayload = Object.freeze({
  contract: 'canonical_results_v1',
  data_source: 'canonical',
  authority: 'canonical_production',
  count: 1,
  items: Object.freeze([
    Object.freeze({
      state: 'ZZ',
      year: 2024,
      election_year: 2024,
      contest: 'Contract Contest',
      jurisdiction_name: 'District 5',
      jurisdiction_type: 'district',
      precinct: null,
      candidate: 'Zero Candidate',
      total_votes: 0,
    }),
  ]),
  semantic_contract: Object.freeze({
    null: 'preserved_null',
    zero: 'numeric_zero',
    null_reason: 'not_inferred',
    no_warehouse_fallback: true,
  }),
});

describe('F2-F results workspace contracts', () => {
  it('renders NULL, numeric zero, and missing cells distinctly', () => {
    expect(formatResultCell(null)).toBe('NULL');
    expect(formatResultCell(0)).toBe('0');
    expect(formatResultCell(undefined)).toBe('MISSING');

    const output = runtimeResult.outputs[0];
    expect(filterPreviewRows(output, 'null')).toHaveLength(1);
    expect(filterPreviewRows(output, 'zero candidate')).toHaveLength(1);
    expect(filterPreviewRows(output, 'not present')).toHaveLength(0);
    expect(output.rows[0].precinct).toBeNull();
    expect(output.rows[0].total_votes).toBe(0);
  });

  it('builds only a same-origin GET read path from safe source metadata', () => {
    expect(buildCanonicalReadPath('/api/ballotlens-database', source)).toBe(
      '/api/ballotlens-database?state=ZZ&year=2024&contest=Contract+Contest&limit=1000',
    );
    expect(() => buildCanonicalReadPath('https://example.invalid/data', source))
      .toThrow(/same-origin relative/);
    expect(() => buildCanonicalReadPath('//example.invalid/data', source))
      .toThrow(/same-origin relative/);
    expect(() => buildCanonicalReadPath('/api/warehouse_election_results', source))
      .toThrow(/approved production read surface/);
  });

  it('parses canonical authority without coercing null or zero', () => {
    const envelope = parseCanonicalEnvelope(canonicalPayload);
    expect(envelope.authority).toBe('canonical_production');
    expect(envelope.items[0].precinct).toBeNull();
    expect(envelope.items[0].total_votes).toBe(0);
  });

  it('reports exact match only for deterministic row equality', () => {
    const envelope = parseCanonicalEnvelope(canonicalPayload);
    expect(compareRuntimeToCanonical(runtimeResult, envelope)).toEqual({
      outcome: 'EXACT_MATCH',
      reason: 'exact_deterministic_row_match',
      canonical_count: 1,
      parser_row_count: 1,
    });

    const different = parseCanonicalEnvelope({
      ...canonicalPayload,
      items: [{ ...canonicalPayload.items[0], total_votes: 1 }],
    });
    expect(compareRuntimeToCanonical(runtimeResult, different)).toMatchObject({
      outcome: 'UNRESOLVED',
      reason: 'row_values_not_exact',
    });
  });

  it('keeps non-comparable parser evidence unresolved rather than mismatched', () => {
    const wideResult: PublicRuntimeResult = Object.freeze({
      ...runtimeResult,
      outputs: Object.freeze([
        Object.freeze({
          ...runtimeResult.outputs[0],
          headers: Object.freeze(['Precinct', 'Candidate - Total Votes']),
          rows: Object.freeze([
            Object.freeze({
              Precinct: null,
              'Candidate - Total Votes': 0,
            }),
          ]),
        }),
      ]),
    });
    const comparison = compareRuntimeToCanonical(
      wideResult,
      parseCanonicalEnvelope(canonicalPayload),
    );
    expect(comparison.outcome).toBe('UNRESOLVED');
    expect(comparison.reason).toBe(
      'parser_shape_not_deterministically_comparable',
    );
  });

  it('performs canonical comparison with GET and same-origin credentials only', async () => {
    let observedInput = '';
    let observedInit: RequestInit | undefined;
    const fetchImpl: typeof fetch = async (input, init) => {
      observedInput = String(input);
      observedInit = init;
      return new Response(JSON.stringify(canonicalPayload), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    };

    const comparison = await fetchCanonicalComparison(
      '/api/ballotlens-database',
      runtimeResult,
      fetchImpl,
    );

    expect(observedInput).toContain('/api/ballotlens-database?');
    expect(observedInit?.method).toBe('GET');
    expect(observedInit?.credentials).toBe('same-origin');
    expect(comparison.outcome).toBe('EXACT_MATCH');
  });
});
