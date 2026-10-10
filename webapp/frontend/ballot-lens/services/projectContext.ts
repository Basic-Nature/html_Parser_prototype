/** Query project IDs are untrusted selectors; only a 200 server response permits a return link. */
export const PROJECT_UUID_RX = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
export function readProjectQuerySelector(): string | null {
  if (typeof window === 'undefined') return null;
  const value = new URLSearchParams(window.location.search).get('project_id');
  return value && PROJECT_UUID_RX.test(value) ? value.toLowerCase() : null;
}
export async function resolveProjectReturn(selector: string, signal?: AbortSignal): Promise<string|null> {
  if (!PROJECT_UUID_RX.test(selector)) return null;
  try {
    const response = await fetch(`/api/projects/v1/${encodeURIComponent(selector)}`, {
      credentials: 'same-origin', cache: 'no-store', signal,
      headers: {Accept: 'application/json'},
    });
    if (!response.ok) return null;
    const payload: unknown = await response.json();
    if (!payload || typeof payload !== 'object' || !('id' in payload)) return null;
    const id = (payload as {id?: unknown}).id;
    return typeof id === 'string' && id.toLowerCase() === selector.toLowerCase()
      ? `/projects/${encodeURIComponent(selector)}` : null;
  } catch {return null;}
}
