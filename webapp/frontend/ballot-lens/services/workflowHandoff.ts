import type { TrustedSourceSelection } from './trustedExecution';

export type WorkflowHandoffSelection = Extract<
  TrustedSourceSelection,
  { readonly runMode: 'worklist' }
>;

export interface WorkflowHandoffQueryIntent {
  readonly present: boolean;
  readonly workflowItemId: string | null;
}

export const WORKFLOW_HANDOFF_QUERY_KEYS = Object.freeze([
  'workflow_item_id',
] as const);

export const WORKFLOW_HANDOFF_QUERY_CLEANUP_KEYS = Object.freeze([
  'workflow_item_id',
  'workflow_pass_id',
  'expected_row_version',
] as const);

const WORKFLOW_HANDOFF_BROWSER_RESPONSE_KEYS = Object.freeze([
  'browser_payload_keys',
  'can_execute_ballot_lens',
  'contract',
  'expected_row_version',
  'principal_disclosed',
  'source_url_disclosed',
  'success',
  'workflow_item_id',
  'workflow_pass_id',
] as const);

const WORKFLOW_HANDOFF_BROWSER_PAYLOAD_KEYS = Object.freeze([
  'workflow_item_id',
  'workflow_pass_id',
  'expected_row_version',
] as const);

const WORKFLOW_HANDOFF_UUID_RX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function normalizedUuid(value: string | null): string | null {
  const candidate = value?.trim() ?? '';
  return WORKFLOW_HANDOFF_UUID_RX.test(candidate) ? candidate.toLowerCase() : null;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function normalizedResolvedSelection(
  payload: unknown,
  requestedWorkflowItemId: string,
): WorkflowHandoffSelection | null {
  const value = record(payload);
  if (!value) return null;

  const responseKeys = Object.keys(value).sort();
  const requiredKeys = [...WORKFLOW_HANDOFF_BROWSER_RESPONSE_KEYS].sort();
  if (
    responseKeys.length !== requiredKeys.length
    || !requiredKeys.every((key, index) => key === responseKeys[index])
  ) return null;

  if (
    value.success !== true
    || value.contract !== 'workflow_ballot_lens_handoff_v1'
    || value.can_execute_ballot_lens !== true
    || value.principal_disclosed !== false
    || value.source_url_disclosed !== false
  ) return null;

  const browserPayloadKeys = value.browser_payload_keys;
  if (
    !Array.isArray(browserPayloadKeys)
    || browserPayloadKeys.length !== WORKFLOW_HANDOFF_BROWSER_PAYLOAD_KEYS.length
    || !WORKFLOW_HANDOFF_BROWSER_PAYLOAD_KEYS.every(
      (key, index) => browserPayloadKeys[index] === key,
    )
  ) return null;

  const workflowItemId = normalizedUuid(
    typeof value.workflow_item_id === 'string' ? value.workflow_item_id : null,
  );
  const workflowPassId = normalizedUuid(
    typeof value.workflow_pass_id === 'string' ? value.workflow_pass_id : null,
  );
  const expectedRowVersion = value.expected_row_version;
  if (
    workflowItemId !== requestedWorkflowItemId
    || !workflowPassId
    || !Number.isSafeInteger(expectedRowVersion)
    || (expectedRowVersion as number) <= 0
  ) return null;

  return Object.freeze({
    runMode: 'worklist' as const,
    displayLabel: 'Governed Workflow task',
    workflowItemId,
    workflowPassId,
    expectedRowVersion: expectedRowVersion as number,
  });
}


export function readWorkflowHandoffQueryIntent(
  raw: string = (
    typeof window === 'undefined' ? '' : window.location.search
  ),
): WorkflowHandoffQueryIntent {
  const params = new URLSearchParams(raw);
  return {
    present: params.has('workflow_item_id'),
    workflowItemId: normalizedUuid(params.get('workflow_item_id')),
  };
}

export async function resolveWorkflowHandoffSelection(
  workflowItemId: string,
): Promise<WorkflowHandoffSelection | null> {
  const normalized = normalizedUuid(workflowItemId);
  if (!normalized) return null;
  let response: Response;
  try {
    response = await fetch(
      `/api/workflow/v1/contributor/items/${encodeURIComponent(normalized)}/ballot-lens-handoff`,
      {
        method: 'GET',
        credentials: 'same-origin',
        headers: { Accept: 'application/json' },
      },
    );
  } catch {
    return null;
  }
  if (!response.ok) return null;
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    return null;
  }
  return normalizedResolvedSelection(payload, normalized);
}

export function clearWorkflowHandoffQuery(): void {
  if (
    typeof window === 'undefined'
    || !window.history?.replaceState
  ) return;
  const url = new URL(window.location.href);
  for (const key of WORKFLOW_HANDOFF_QUERY_CLEANUP_KEYS) {
    url.searchParams.delete(key);
  }
  const nextLocation = `${url.pathname}${url.search}${url.hash}`;
  const currentLocation =
    `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextLocation !== currentLocation) {
    window.history.replaceState(window.history.state, '', nextLocation);
  }
}
