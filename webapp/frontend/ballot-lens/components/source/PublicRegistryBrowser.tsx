import {
  useEffect,
  useMemo,
  useState,
} from 'react';

import {
  EMPTY_REGISTRY_FILTERS,
  filterRegistrySources,
  getRegistryFacetOptions,
  registrySourceLabel,
  type PublicRegistryEnvelope,
  type PublicRegistrySource,
  type RegistryFacetKey,
  type RegistryFilters,
} from '../../contracts/registry';
import { loadPublicRegistryEnvelope } from '../../services/registryApi';

interface PublicRegistryBrowserProps {
  readonly endpoint: string;
  readonly selectedSourceId: string;
  readonly selectionLocked: boolean;
  readonly onRegistryEnvelopeChange: (
    envelope: PublicRegistryEnvelope | null,
  ) => void;
  readonly onSelectionChange: (source: PublicRegistrySource | null) => void;
}

type LoadState = 'loading' | 'ready' | 'empty' | 'error';

const FACETS: readonly {
  readonly key: RegistryFacetKey;
  readonly label: string;
}[] = [
  { key: 'state', label: 'State' },
  { key: 'year', label: 'Year' },
  { key: 'contest', label: 'Contest' },
  { key: 'scope', label: 'Scope / county' },
  { key: 'format', label: 'Format' },
];

export function PublicRegistryBrowser({
  endpoint,
  selectedSourceId,
  selectionLocked,
  onRegistryEnvelopeChange,
  onSelectionChange,
}: PublicRegistryBrowserProps) {
  const [sources, setSources] = useState<readonly PublicRegistrySource[]>([]);
  const [filters, setFilters] =
    useState<RegistryFilters>(EMPTY_REGISTRY_FILTERS);
  const [loadState, setLoadState] = useState<LoadState>('loading');

  useEffect(() => {
    const controller = new AbortController();
    setLoadState('loading');

    loadPublicRegistryEnvelope(endpoint, controller.signal)
      .then((envelope) => {
        setSources(envelope.sources);
        onRegistryEnvelopeChange(envelope);
        setLoadState(envelope.sources.length ? 'ready' : 'empty');
      })
      .catch((error: unknown) => {
        if (
          error instanceof DOMException
          && error.name === 'AbortError'
        ) {
          return;
        }
        setSources([]);
        onRegistryEnvelopeChange(null);
        onSelectionChange(null);
        setLoadState('error');
      });

    return () => controller.abort();
  }, [endpoint, onRegistryEnvelopeChange, onSelectionChange]);

  const visibleSources = useMemo(
    () => filterRegistrySources(sources, filters),
    [sources, filters],
  );

  const selected = sources.find(
    (source) => source.registry_source_id === selectedSourceId,
  ) ?? null;

  const updateFilter = (
    key: keyof RegistryFilters,
    value: string,
  ) => {
    setFilters((current) => ({
      ...current,
      [key]: value,
    }));
  };

  return (
    <section className="blf2-registry-browser" aria-label="Approved public registry">
      <div className="blf2-registry-head">
        <div>
          <span className="blf2-kicker">Approved sources</span>
          <strong>Public Registry</strong>
        </div>
        <span className="blf2-registry-count">
          {loadState === 'ready'
            ? `${visibleSources.length} / ${sources.length}`
            : loadState}
        </span>
      </div>

      <label className="blf2-registry-search">
        <span>Search approved sources</span>
        <input
          type="search"
          value={filters.query}
          onChange={(event) => updateFilter('query', event.target.value)}
          placeholder="State, contest, scope, format…"
          disabled={loadState !== 'ready'}
        />
      </label>

      <div className="blf2-registry-facets" aria-label="Registry filters">
        {FACETS.map(({ key, label }) => {
          const options = getRegistryFacetOptions(sources, filters, key);
          return (
            <label key={key}>
              <span>{label}</span>
              <select
                value={filters[key]}
                onChange={(event) => updateFilter(key, event.target.value)}
                disabled={loadState !== 'ready'}
              >
                <option value="">All</option>
                {options.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    disabled={!option.available && filters[key] !== option.value}
                  >
                    {option.value} ({option.count})
                  </option>
                ))}
              </select>
            </label>
          );
        })}
      </div>

      <div className="blf2-registry-status" role="status">
        {loadState === 'loading' && 'Loading approved source metadata…'}
        {loadState === 'error' && 'Approved public registry could not be loaded.'}
        {loadState === 'empty' && 'No approved public registry sources are available.'}
        {loadState === 'ready' && visibleSources.length === 0
          && 'No approved sources match the current filters.'}
        {loadState === 'ready' && visibleSources.length > 0
          && 'Select the one execution-authorized source to enable submission.'}
      </div>

      {loadState === 'ready' && visibleSources.length > 0 && (
        <div className="blf2-registry-list" aria-label="Approved source results">
          {visibleSources.map((source) => (
            <button
              key={source.registry_source_id}
              type="button"
              className="blf2-registry-source"
              data-selected={
                source.registry_source_id === selectedSourceId ? 'true' : 'false'
              }
              disabled={selectionLocked}
              onClick={() => onSelectionChange(source)}
            >
              <strong>{registrySourceLabel(source)}</strong>
              <small>
                Curated registry metadata • no executable URL projected
              </small>
            </button>
          ))}
        </div>
      )}

      <div className="blf2-registry-selection" aria-label="Selected approved source">
        <span className="blf2-kicker">Selection</span>
        {!selected ? (
          <>
            <strong>No approved source selected</strong>
            <p>Select a registry entry to review its safe public metadata.</p>
          </>
        ) : (
          <>
            <strong>{registrySourceLabel(selected)}</strong>
            <dl>
              <div><dt>Year</dt><dd>{selected.year || '—'}</dd></div>
              <div><dt>State</dt><dd>{selected.state || '—'}</dd></div>
              <div><dt>Contest</dt><dd>{selected.contest || '—'}</dd></div>
              <div><dt>Scope</dt><dd>{selected.scope || '—'}</dd></div>
              <div><dt>Format</dt><dd>{selected.format || '—'}</dd></div>
              <div><dt>Registry</dt><dd>Curated</dd></div>
            </dl>
            <p>
              Selection is app-owned. Execution remains limited to the exact
              registry authority projected by the server.
            </p>
          </>
        )}
      </div>
    </section>
  );
}
