import { describe, expect, it } from 'vitest';

import {
  parsePublicMemoryPreview,
  parsePublicRuntimeResult,
} from '../contracts/publicRuntime';

const source = {
  contest: 'President',
  format: 'HTML',
  registry_category: 'curated',
  registry_source_id: 'source-ut',
  scope: 'statewide',
  state: 'Utah',
  year: '2024',
};

const preview = {
  contract: 'ballot_lens_public_memory_preview_v1',
  registry_source_id: 'source-ut',
  source,
  headers: ['Precinct', 'Candidate - Total Votes'],
  rows: [
    {
      Precinct: null,
      'Candidate - Total Votes': 42,
    },
  ],
  row_count: 1,
  output_mode: 'MEMORY_PREVIEW_ONLY',
  download_available: false,
  persistent_output: false,
  execution_context_contract: 'ballot_lens_public_execution_context_v1',
  progress: [
    {
      type: 'run_progress',
      processed: 1,
      total_entries: 1,
      status_counts: { success: 1 },
    },
  ],
};

const result = {
  contract: 'ballot_lens_public_runtime_result_v1',
  registry_source_id: 'source-ut',
  source,
  outputs: [preview],
  status_counts: { success: 1 },
  terminal_status: 'success',
  terminal_reason_code: null,
  download_available: false,
  persistent_output: false,
};

describe('public runtime contracts', () => {
  it('preserves semantic null in bounded memory previews', () => {
    const parsed = parsePublicMemoryPreview(preview);
    expect(parsed.rows[0]?.Precinct).toBeNull();
    expect(parsed.download_available).toBe(false);
    expect(parsed.persistent_output).toBe(false);
    expect(parsed.output_mode).toBe('MEMORY_PREVIEW_ONLY');
  });

  it('validates terminal result ownership and no-download policy', () => {
    const parsed = parsePublicRuntimeResult(result);
    expect(parsed.registry_source_id).toBe('source-ut');
    expect(parsed.outputs).toHaveLength(1);
    expect(parsed.status_counts.success).toBe(1);
    expect(parsed.download_available).toBe(false);
  });

  it('rejects persistence or download drift', () => {
    expect(() => parsePublicRuntimeResult({
      ...result,
      download_available: true,
    })).toThrow(/persistence\/download policy drift/);
  });

  it('rejects output source mismatches', () => {
    expect(() => parsePublicRuntimeResult({
      ...result,
      outputs: [
        {
          ...preview,
          registry_source_id: 'source-other',
        },
      ],
    })).toThrow(/source mismatch/);
  });

  it('rejects unallowlisted terminal reason codes', () => {
    expect(() => parsePublicRuntimeResult({
      ...result,
      terminal_reason_code: 'unexpected_reason',
    })).toThrow(/Unallowlisted public terminal reason/);
  });
});
