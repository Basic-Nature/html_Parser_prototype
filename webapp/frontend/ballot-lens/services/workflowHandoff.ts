import type { TrustedSourceSelection } from './trustedExecution';

export type WorkflowHandoffSelection = Extract<
  TrustedSourceSelection,
  { readonly runMode: 'worklist' }
>;

export interface WorkflowHandoffQueryIntent {
  readonly present: boolean;
  readonly selection: WorkflowHandoffSelection | null;
}

export const WORKFLOW_HANDOFF_QUERY_KEYS = Object.freeze([
  'workflow_item_id',
  'workflow_pass_id',
  'expected_row_version',
] as const);

const UUID_RX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function normalizedUuid(value: string | null): string | null {
  const candidate = value?.trim() ?? '';
  return UUID_RX.test(candidate) ? candidate.toLowerCase() : null;
}

function normalizedRowVersion(value: string | null): number | null {
  const candidate = value?.trim() ?? '';
  if (!/^[1-9][0-9]*$/.test(candidate)) return null;
  const parsed = Number(candidate);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

export function readWorkflowHandoffQueryIntent(
  search?: string,
): WorkflowHandoffQueryIntent {
  const raw = search ?? (
    typeof window === 'undefined' ? '' : window.location.search
  );
  const params = new URLSearchParams(raw);
  const present = WORKFLOW_HANDOFF_QUERY_KEYS.some(
    key => params.has(key),
  );
  if (!present) {
    return Object.freeze({ present: false, selection: null });
  }

  const workflowItemId = normalizedUuid(
    params.get('workflow_item_id'),
  );
  const workflowPassId = normalizedUuid(
    params.get('workflow_pass_id'),
  );
  const expectedRowVersion = normalizedRowVersion(
    params.get('expected_row_version'),
  );
  if (!workflowItemId || !workflowPassId || !expectedRowVersion) {
    return Object.freeze({ present: true, selection: null });
  }

  return Object.freeze({
    present: true,
    selection: Object.freeze({
      runMode: 'worklist' as const,
      displayLabel: 'Governed Workflow task',
      workflowItemId,
      workflowPassId,
      expectedRowVersion,
    }),
  });
}

export function clearWorkflowHandoffQuery(): void {
  if (
    typeof window === 'undefined'
    || !window.history?.replaceState
  ) {
    return;
  }

  const url = new URL(window.location.href);
  for (const key of WORKFLOW_HANDOFF_QUERY_KEYS) {
    url.searchParams.delete(key);
  }
  const next = `${url.pathname}${url.search}${url.hash}`;
  const current =
    `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) {
    window.history.replaceState(window.history.state, '', next);
  }
}
