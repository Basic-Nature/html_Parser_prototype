import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  WORKFLOW_HANDOFF_QUERY_CLEANUP_KEYS,
  WORKFLOW_HANDOFF_QUERY_KEYS,
  clearWorkflowHandoffQuery,
  readWorkflowHandoffQueryIntent,
  resolveWorkflowHandoffSelection,
} from '../services/workflowHandoff';

const ITEM = '11111111-1111-1111-1111-111111111111';
const PASS = '22222222-2222-2222-2222-222222222222';
const UPPER_ITEM = 'ABCDEFAB-CDEF-ABCD-EFAB-CDEFABCDEFAB';
const LOWER_ITEM = UPPER_ITEM.toLowerCase();

function browserHandoff(overrides: Record<string, unknown> = {}) {
  return {
    browser_payload_keys: [
      'workflow_item_id',
      'workflow_pass_id',
      'expected_row_version',
    ],
    can_execute_ballot_lens: true,
    contract: 'workflow_ballot_lens_handoff_v1',
    expected_row_version: 7,
    principal_disclosed: false,
    source_url_disclosed: false,
    success: true,
    workflow_item_id: ITEM,
    workflow_pass_id: PASS,
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('O4D Workflow handoff selector-only query intent', () => {
  it('accepts only workflow_item_id as navigation context', () => {
    expect(WORKFLOW_HANDOFF_QUERY_KEYS).toEqual(['workflow_item_id']);
    expect(readWorkflowHandoffQueryIntent(
      `?workflow_item_id=${ITEM}&workflow_pass_id=${PASS}&expected_row_version=7`,
    )).toEqual({ present: true, workflowItemId: ITEM });
  });

  it('preserves generic canonical UUID compatibility and lowercase normalization', () => {
    expect(readWorkflowHandoffQueryIntent(
      `?workflow_item_id=${UPPER_ITEM}`,
    )).toEqual({ present: true, workflowItemId: LOWER_ITEM });
    expect(readWorkflowHandoffQueryIntent(
      `?workflow_item_id=bad&workflow_pass_id=${PASS}&expected_row_version=7`,
    )).toEqual({ present: true, workflowItemId: null });
    expect(readWorkflowHandoffQueryIntent(
      `?workflow_pass_id=${PASS}&expected_row_version=7`,
    )).toEqual({ present: false, workflowItemId: null });
  });

  it('clears selector and legacy non-authority keys while preserving unrelated URL state', () => {
    const registrySourceId = `blsrc_v1_${'a'.repeat(64)}`;
    const search = (
      `?workflow_item_id=${ITEM}`
      + `&workflow_pass_id=${PASS}`
      + '&expected_row_version=7'
      + `&source=${registrySourceId}`
    );
    const replaceState = vi.fn();
    const state = { retained: true };
    vi.stubGlobal('window', {
      location: {
        href: `https://electionpulse.test/ballot_lens${search}#inspect`,
        pathname: '/ballot_lens',
        search,
        hash: '#inspect',
      },
      history: {
        state,
        replaceState,
      },
    });

    expect(WORKFLOW_HANDOFF_QUERY_KEYS).toEqual(['workflow_item_id']);
    expect(WORKFLOW_HANDOFF_QUERY_CLEANUP_KEYS).toEqual([
      'workflow_item_id',
      'workflow_pass_id',
      'expected_row_version',
    ]);

    clearWorkflowHandoffQuery();

    expect(replaceState).toHaveBeenCalledTimes(1);
    expect(replaceState).toHaveBeenCalledWith(
      state,
      '',
      `/ballot_lens?source=${registrySourceId}#inspect`,
    );
  });
});

describe('O4D target-side Workflow handoff resolution', () => {
  it('uses the dedicated item-keyed same-origin GET and exact browser contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => browserHandoff(),
    });
    vi.stubGlobal('fetch', fetchMock);

    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toEqual({
      runMode: 'worklist',
      displayLabel: 'Governed Workflow task',
      workflowItemId: ITEM,
      workflowPassId: PASS,
      expectedRowVersion: 7,
    });
    expect(fetchMock).toHaveBeenCalledWith(
      `/api/workflow/v1/contributor/items/${ITEM}/ballot-lens-handoff`,
      { method: 'GET', credentials: 'same-origin', headers: { Accept: 'application/json' } },
    );
  });

  it('fails closed for restricted, malformed, mismatched, or non-browser state', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({}),
    }));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => browserHandoff({ workflow_pass_id: 'bad' }),
    }));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => browserHandoff({
        workflow_item_id: '33333333-3333-3333-3333-333333333333',
      }),
    }));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => browserHandoff({ execution_mode: 'workflow' }),
    }));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => browserHandoff({
        browser_payload_keys: ['workflow_item_id', 'workflow_pass_id'],
      }),
    }));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();

    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('unavailable')));
    await expect(resolveWorkflowHandoffSelection(ITEM)).resolves.toBeNull();
  });
});
