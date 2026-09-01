import { describe, expect, it } from 'vitest';

import {
  EMPTY_REGISTRY_FILTERS,
  filterRegistrySources,
  getRegistryFacetOptions,
  parsePublicRegistryPayload,
} from '../contracts/registry';

const payload = {
  contract: 'ballot_lens_public_registry_v1',
  sources: [
    {
      contest: 'President',
      format: 'HTML',
      registry_category: 'curated',
      registry_source_id: 'source-ut',
      scope: 'statewide',
      state: 'Utah',
      year: 2024,
    },
    {
      contest: 'President',
      format: 'CSV',
      registry_category: 'curated',
      registry_source_id: 'source-az',
      scope: 'statewide',
      state: 'Arizona',
      year: 2024,
    },
    {
      contest: 'General Election',
      format: 'HTML',
      registry_category: 'curated',
      registry_source_id: 'source-ny',
      scope: 'Rockland',
      state: 'New York',
      year: 2024,
    },
  ],
};

describe('public registry discovery contract', () => {
  it('normalizes the exact safe public projection', () => {
    const sources = parsePublicRegistryPayload(payload);
    expect(sources).toHaveLength(3);
    expect(sources[0]?.year).toBe('2024');
    expect(Object.keys(sources[0] ?? {}).sort()).toEqual([
      'contest',
      'format',
      'registry_category',
      'registry_source_id',
      'scope',
      'state',
      'year',
    ]);
  });

  it('fails closed if an executable URL-like field leaks into a source', () => {
    const unsafe = {
      ...payload,
      sources: [
        {
          ...payload.sources[0],
          url: 'https://example.invalid/results',
        },
      ],
    };
    expect(() => parsePublicRegistryPayload(unsafe)).toThrow(
      /Unsafe public registry source projection/,
    );
  });

  it('rejects non-curated public entries', () => {
    const invalid = {
      ...payload,
      sources: [
        {
          ...payload.sources[0],
          registry_category: 'unreviewed',
        },
      ],
    };
    expect(() => parsePublicRegistryPayload(invalid)).toThrow(
      /Non-curated registry source/,
    );
  });

  it('filters by search and structured facets without URLs', () => {
    const sources = parsePublicRegistryPayload(payload);
    const filtered = filterRegistrySources(sources, {
      ...EMPTY_REGISTRY_FILTERS,
      query: 'President',
      state: 'Utah',
    });
    expect(filtered.map((source) => source.registry_source_id)).toEqual([
      'source-ut',
    ]);
  });

  it('keeps unavailable facet values visible with zero counts', () => {
    const sources = parsePublicRegistryPayload(payload);
    const options = getRegistryFacetOptions(
      sources,
      {
        ...EMPTY_REGISTRY_FILTERS,
        state: 'Utah',
      },
      'format',
    );

    expect(options).toContainEqual({
      value: 'CSV',
      count: 0,
      available: false,
    });
    expect(options).toContainEqual({
      value: 'HTML',
      count: 1,
      available: true,
    });
  });
});
