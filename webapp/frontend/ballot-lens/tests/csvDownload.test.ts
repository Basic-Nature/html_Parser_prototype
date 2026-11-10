import { test, expect } from 'vitest';
import { parsePublicCsvOffer } from '../contracts/csvDownload';
const offer = { contract: 'ballot_lens_public_csv_offer_v1', registry_source_id: 'blsrc_v1_' + 'a'.repeat(64), token: 'X'.repeat(43), endpoint: '/api/public/csv-download', expires_in_seconds: 120 };
test('accepts only the exact one-use public CSV event', () => {
  expect(parsePublicCsvOffer(offer)).not.toBeNull();
  expect(parsePublicCsvOffer({...offer, endpoint: 'https://evil.example/download'})).toBeNull();
  expect(parsePublicCsvOffer({...offer, token: 'short'})).toBeNull();
  expect(parsePublicCsvOffer({...offer, payload: 'extra'})).toBeNull();
  expect(parsePublicCsvOffer({...offer, registry_source_id: 'url:https://bad.example'})).toBeNull();
});
