import { describe, expect, it } from 'vitest';

import {
  WORKFLOW_HANDOFF_QUERY_KEYS,
  readWorkflowHandoffQueryIntent,
} from '../services/workflowHandoff';

describe('W13D Workflow handoff query intent', () => {
  it('accepts only the three governed identifier values', () => {
    const item = '11111111-1111-1111-1111-111111111111';
    const pass = '22222222-2222-2222-2222-222222222222';
    const intent = readWorkflowHandoffQueryIntent(
      `?workflow_item_id=${item}&workflow_pass_id=${pass}&expected_row_version=7`,
    );

    expect(intent.present).toBe(true);
    expect(intent.selection).toEqual({
      runMode: 'worklist',
      displayLabel: 'Governed Workflow task',
      workflowItemId: item,
      workflowPassId: pass,
      expectedRowVersion: 7,
    });
    expect(WORKFLOW_HANDOFF_QUERY_KEYS).toEqual([
      'workflow_item_id',
      'workflow_pass_id',
      'expected_row_version',
    ]);
  });

  it('fails closed on partial or malformed Workflow intent', () => {
    expect(
      readWorkflowHandoffQueryIntent(
        '?workflow_item_id=11111111-1111-1111-1111-111111111111',
      ),
    ).toEqual({ present: true, selection: null });

    expect(
      readWorkflowHandoffQueryIntent(
        '?workflow_item_id=bad&workflow_pass_id=22222222-2222-2222-2222-222222222222&expected_row_version=7',
      ),
    ).toEqual({ present: true, selection: null });

    expect(
      readWorkflowHandoffQueryIntent(
        '?workflow_item_id=11111111-1111-1111-1111-111111111111&workflow_pass_id=22222222-2222-2222-2222-222222222222&expected_row_version=0',
      ),
    ).toEqual({ present: true, selection: null });
  });

  it('does not manufacture Workflow intent when absent', () => {
    expect(readWorkflowHandoffQueryIntent('?source=blsrc_v1_deadbeef'))
      .toEqual({ present: false, selection: null });
  });
});
