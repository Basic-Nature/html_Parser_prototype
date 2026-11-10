/** One-time same-origin CSV capability. Never consumes client-supplied election rows. */
export interface PublicCsvOffer {
  readonly contract: 'ballot_lens_public_csv_offer_v1';
  readonly registry_source_id: string;
  readonly token: string;
  readonly endpoint: '/api/public/csv-download';
  readonly expires_in_seconds: 120;
}

const sourceIdRx = /^blsrc_v1_[a-f0-9]{64}$/;
const tokenRx = /^[A-Za-z0-9_-]{43}$/;

export function parsePublicCsvOffer(raw: unknown): PublicCsvOffer | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  if (Object.keys(o).sort().join('|') !==
      'contract|endpoint|expires_in_seconds|registry_source_id|token') return null;
  if (o.contract !== 'ballot_lens_public_csv_offer_v1'
    || typeof o.registry_source_id !== 'string'
    || !sourceIdRx.test(o.registry_source_id)
    || typeof o.token !== 'string' || !tokenRx.test(o.token)
    || o.endpoint !== '/api/public/csv-download'
    || o.expires_in_seconds !== 120) return null;
  return o as unknown as PublicCsvOffer;
}

export async function consumePublicCsvOffer(offer: PublicCsvOffer): Promise<Blob> {
  const response = await fetch(offer.endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/csv' },
    body: JSON.stringify({ token: offer.token }),
    cache: 'no-store',
    credentials: 'same-origin',
    redirect: 'error',
  });
  if (!response.ok || !response.headers.get('Content-Type')?.startsWith('text/csv')) {
    throw new Error('The one-use CSV is unavailable or expired. Run the approved source again.');
  }
  return response.blob();
}
