import { describe, expect, it } from 'vitest';

import {
  EMPTY_REGISTRY_FILTERS,
  filterRegistrySources,
  getRegistryFacetOptions,
  parsePublicRegistryPayload,
} from '../contracts/registry';

const payload = {
  contract: 'ballot_lens_public_registry_v1',
  count: 3,
  execution_enabled: true,
  execution_source_id: 'source-ut',
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
  it('normalizes root execution authority without changing source shape', () => {
    const envelope = parsePublicRegistryPayload(payload);
    expect(envelope.execution_enabled).toBe(true);
    expect(envelope.execution_source_id).toBe('source-ut');
    expect(envelope.count).toBe(3);
    expect(envelope.sources).toHaveLength(3);
    expect(envelope.sources[0]?.year).toBe('2024');
    expect(Object.keys(envelope.sources[0] ?? {}).sort()).toEqual([
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
      count: 1,
    };
    expect(() => parsePublicRegistryPayload(unsafe)).toThrow(
      /Unsafe public registry source projection/,
    );
  });

  it('rejects unexpected root fields instead of inheriting authority', () => {
    const unsafe = {
      ...payload,
      executable_url: 'https://example.invalid/results',
    };
    expect(() => parsePublicRegistryPayload(unsafe)).toThrow(
      /Unsafe public registry root projection/,
    );
  });

  it('rejects execution ids absent from safe projected sources', () => {
    const invalid = {
      ...payload,
      execution_source_id: 'source-missing',
    };
    expect(() => parsePublicRegistryPayload(invalid)).toThrow(
      /absent from safe registry sources/,
    );
  });

  it('requires disabled execution to project no execution id', () => {
    const disabled = {
      ...payload,
      execution_enabled: false,
      execution_source_id: null,
    };
    const envelope = parsePublicRegistryPayload(disabled);
    expect(envelope.execution_enabled).toBe(false);
    expect(envelope.execution_source_id).toBeNull();
  });

  it('rejects non-curated public entries', () => {
    const invalid = {
      ...payload,
      count: 1,
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
    const sources = parsePublicRegistryPayload(payload).sources;
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
    const sources = parsePublicRegistryPayload(payload).sources;
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
