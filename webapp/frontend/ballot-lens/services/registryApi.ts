import {
  parsePublicRegistryPayload,
  type PublicRegistrySource,
} from '../contracts/registry';

function requireSameOriginRelativeEndpoint(endpoint: string): string {
  const normalized = endpoint.trim();
  if (!normalized.startsWith('/') || normalized.startsWith('//')) {
    throw new Error('Public registry endpoint must be same-origin relative');
  }
  return normalized;
}

export async function loadPublicRegistry(
  endpoint: string,
  signal?: AbortSignal,
): Promise<readonly PublicRegistrySource[]> {
  const safeEndpoint = requireSameOriginRelativeEndpoint(endpoint);
  const response = await fetch(safeEndpoint, {
    method: 'GET',
    credentials: 'same-origin',
    headers: {
      Accept: 'application/json',
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Public registry unavailable (${response.status})`);
  }

  const payload: unknown = await response.json();
  return parsePublicRegistryPayload(payload);
}
