import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
} from 'react';

import type {
  JsonValue,
  PublicMemoryPreview,
  PublicRuntimeResult,
} from '../../contracts/publicRuntime';
import {
  registrySourceLabel,
  type PublicRegistrySource,
} from '../../contracts/registry';
import {
  fetchCanonicalComparison,
  type CanonicalComparison,
} from '../../services/canonicalComparison';

export type WorkspaceTab =
  | 'Results'
  | 'Validation'
  | 'Metadata'
  | 'Provenance';

interface WorkspaceViewsProps {
  readonly activeTab: WorkspaceTab;
  readonly selectedSource: PublicRegistrySource | null;
  readonly runtimeResult: PublicRuntimeResult | null;
  readonly dataApiUrl: string;
}

interface ComparisonState {
  readonly loading: boolean;
  readonly comparison: CanonicalComparison;
}

function unresolvedComparison(reason: string, rowCount = 0): CanonicalComparison {
  return Object.freeze({
    outcome: 'UNRESOLVED' as const,
    reason,
    canonical_count: 0,
    parser_row_count: rowCount,
  });
}

export function formatResultCell(value: JsonValue | undefined): string {
  if (value === undefined) return 'MISSING';
  if (value === null) return 'NULL';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(value);
}

export function filterPreviewRows(
  output: PublicMemoryPreview,
  query: string,
): readonly Readonly<Record<string, JsonValue>>[] {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return output.rows;
  return output.rows.filter((row) => (
    output.headers.some((header) => (
      formatResultCell(row[header]).toLowerCase().includes(normalized)
    ))
  ));
}

function totalRows(result: PublicRuntimeResult | null): number {
  if (!result) return 0;
  return result.outputs.reduce((total, output) => total + output.row_count, 0);
}

function missingCellCount(output: PublicMemoryPreview): number {
  return output.rows.reduce((total, row) => (
    total + output.headers.reduce((rowTotal, header) => (
      rowTotal + (Object.prototype.hasOwnProperty.call(row, header) ? 0 : 1)
    ), 0)
  ), 0);
}

function FieldList({
  entries,
}: {
  readonly entries: readonly (readonly [string, string])[];
}) {
  return (
    <dl className="blf2-detail-list">
      {entries.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function EmptyView({
  title,
  body,
}: {
  readonly title: string;
  readonly body: string;
}) {
  return (
    <div className="blf2-view-empty" role="status">
      <strong>{title}</strong>
      <p>{body}</p>
    </div>
  );
}

function ResultsView({
  runtimeResult,
}: {
  readonly runtimeResult: PublicRuntimeResult | null;
}) {
  const [query, setQuery] = useState('');

  if (!runtimeResult) {
    return (
      <EmptyView
        title="No owned result to render"
        body="Results are populated only from a real terminal memory result. No sample rows or vote totals are fabricated."
      />
    );
  }

  return (
    <div className="blf2-results-view">
      <div className="blf2-view-toolbar">
        <div>
          <span className="blf2-kicker">Memory-owned result</span>
          <strong>{totalRows(runtimeResult)} rows</strong>
        </div>
        <label className="blf2-result-search">
          <span>Search current result</span>
          <input
            type="search"
            value={query}
            onChange={(event: ChangeEvent<HTMLInputElement>) => (
              setQuery(event.currentTarget.value)
            )}
            placeholder="Search visible values"
          />
        </label>
      </div>

      {runtimeResult.outputs.map((output, outputIndex) => {
        const rows = filterPreviewRows(output, query);
        return (
          <section
            className="blf2-output-block"
            key={`${output.registry_source_id}:${outputIndex}`}
            aria-label={`Memory preview ${outputIndex + 1}`}
          >
            <div className="blf2-output-heading">
              <strong>Memory preview {outputIndex + 1}</strong>
              <span>{rows.length} / {output.row_count} rows shown</span>
            </div>
            <div className="blf2-result-table-wrap">
              <table className="blf2-result-table">
                <thead>
                  <tr>
                    {output.headers.map((header) => (
                      <th key={header} scope="col">{header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {output.headers.map((header) => {
                        const value = row[header];
                        const rendered = formatResultCell(value);
                        return (
                          <td
                            key={header}
                            data-null={value === null ? 'true' : undefined}
                            data-missing={value === undefined ? 'true' : undefined}
                          >
                            {rendered}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={Math.max(1, output.headers.length)}>
                        No rows match the current result search.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}
    </div>
  );
}

function ValidationView({
  runtimeResult,
  comparisonState,
}: {
  readonly runtimeResult: PublicRuntimeResult | null;
  readonly comparisonState: ComparisonState;
}) {
  if (!runtimeResult) {
    return (
      <EmptyView
        title="Validation awaits a terminal result"
        body="Structural and canonical read-only comparison signals appear only after the app owns a real runtime result."
      />
    );
  }

  const missingCells = runtimeResult.outputs.reduce(
    (total, output) => total + missingCellCount(output),
    0,
  );
  const statusCounts = Object.entries(runtimeResult.status_counts);
  const comparison = comparisonState.comparison;

  return (
    <div className="blf2-validation-grid">
      <section className="blf2-detail-card">
        <span className="blf2-kicker">Terminal signal</span>
        <h3>{runtimeResult.terminal_status ?? 'NULL'}</h3>
        <FieldList entries={[
          ['Terminal reason', runtimeResult.terminal_reason_code ?? 'NULL'],
          ['Output count', String(runtimeResult.outputs.length)],
          ['Row count', String(totalRows(runtimeResult))],
          ['Missing cells', String(missingCells)],
        ]} />
      </section>

      <section className="blf2-detail-card">
        <span className="blf2-kicker">Status counts</span>
        <h3>Runtime summary</h3>
        {statusCounts.length ? (
          <FieldList entries={statusCounts.map(([name, count]) => [
            name,
            String(count),
          ] as const)} />
        ) : (
          <p>No terminal status counts were reported.</p>
        )}
      </section>

      <section className="blf2-detail-card blf2-detail-card-wide">
        <span className="blf2-kicker">Canonical comparison</span>
        <h3>{comparisonState.loading ? 'READING' : comparison.outcome}</h3>
        <p>
          Canonical authority is read through the configured GET-only production
          endpoint. Unresolved evidence is never promoted to a mismatch.
        </p>
        <FieldList entries={[
          ['Authority', 'canonical_production'],
          ['Method', 'GET only'],
          ['Outcome', comparisonState.loading ? 'UNRESOLVED' : comparison.outcome],
          ['Reason', comparisonState.loading ? 'canonical_read_in_progress' : comparison.reason],
          ['Canonical rows', String(comparison.canonical_count)],
          ['Parser rows', String(comparison.parser_row_count)],
        ]} />
      </section>
    </div>
  );
}

function MetadataView({
  selectedSource,
  runtimeResult,
}: {
  readonly selectedSource: PublicRegistrySource | null;
  readonly runtimeResult: PublicRuntimeResult | null;
}) {
  const source = runtimeResult?.source ?? selectedSource;
  if (!source) {
    return (
      <EmptyView
        title="No safe source metadata yet"
        body="Select an approved registry source to populate the public metadata projection."
      />
    );
  }

  const runtimeEntries: readonly (readonly [string, string])[] = runtimeResult
    ? [
        ['Runtime contract', runtimeResult.contract],
        ['Registry source ID', runtimeResult.registry_source_id],
        ['Outputs', String(runtimeResult.outputs.length)],
        ['Rows', String(totalRows(runtimeResult))],
        ['Persistent output', String(runtimeResult.persistent_output)],
        ['Download available', String(runtimeResult.download_available)],
      ]
    : [
        ['Runtime contract', 'Awaiting terminal result'],
        ['Registry source ID', source.registry_source_id],
      ];

  return (
    <div className="blf2-validation-grid">
      <section className="blf2-detail-card">
        <span className="blf2-kicker">Safe registry projection</span>
        <h3>{registrySourceLabel(source)}</h3>
        <FieldList entries={[
          ['Registry source ID', source.registry_source_id],
          ['Registry category', source.registry_category],
          ['State', source.state || 'NULL'],
          ['Year', source.year || 'NULL'],
          ['Contest', source.contest || 'NULL'],
          ['Scope', source.scope || 'NULL'],
          ['Format', source.format || 'NULL'],
        ]} />
      </section>
      <section className="blf2-detail-card">
        <span className="blf2-kicker">Runtime metadata</span>
        <h3>Owned result contract</h3>
        <FieldList entries={runtimeEntries} />
      </section>
    </div>
  );
}

function ProvenanceView({
  selectedSource,
  runtimeResult,
  comparisonState,
}: {
  readonly selectedSource: PublicRegistrySource | null;
  readonly runtimeResult: PublicRuntimeResult | null;
  readonly comparisonState: ComparisonState;
}) {
  const source = runtimeResult?.source ?? selectedSource;
  if (!source) {
    return (
      <EmptyView
        title="No provenance record yet"
        body="The public workspace retains only the approved registry projection; raw executable source URLs are not exposed here."
      />
    );
  }

  return (
    <div className="blf2-provenance-stack">
      <section className="blf2-detail-card">
        <span className="blf2-kicker">Parser evidence</span>
        <h3>noncanonical_parser_evidence</h3>
        <FieldList entries={[
          ['Registry source ID', source.registry_source_id],
          ['Source projection', registrySourceLabel(source)],
          ['Raw executable URL', 'Not projected'],
          ['Canonical claim', 'False'],
        ]} />
      </section>

      {runtimeResult?.outputs.map((output, index) => (
        <section
          className="blf2-detail-card"
          key={`${output.registry_source_id}:provenance:${index}`}
        >
          <span className="blf2-kicker">Output {index + 1}</span>
          <h3>{output.contract}</h3>
          <FieldList entries={[
            ['Execution context', output.execution_context_contract],
            ['Output mode', output.output_mode],
            ['Rows', String(output.row_count)],
            ['Persistent output', String(output.persistent_output)],
            ['Download available', String(output.download_available)],
          ]} />
        </section>
      ))}

      {runtimeResult && (
        <section className="blf2-detail-card">
          <span className="blf2-kicker">Canonical read</span>
          <h3>{comparisonState.loading ? 'UNRESOLVED' : comparisonState.comparison.outcome}</h3>
          <FieldList entries={[
            ['Authority', 'canonical_production'],
            ['Method', 'GET only'],
            ['Publication write', 'False'],
            ['Warehouse fallback', 'False'],
            ['Reason', comparisonState.loading
              ? 'canonical_read_in_progress'
              : comparisonState.comparison.reason],
          ]} />
        </section>
      )}
    </div>
  );
}

export function WorkspaceViews({
  activeTab,
  selectedSource,
  runtimeResult,
  dataApiUrl,
}: WorkspaceViewsProps) {
  const rowCount = useMemo(() => totalRows(runtimeResult), [runtimeResult]);
  const [comparisonState, setComparisonState] = useState<ComparisonState>({
    loading: false,
    comparison: unresolvedComparison('canonical_not_requested', rowCount),
  });

  useEffect(() => {
    if (!runtimeResult) {
      setComparisonState({
        loading: false,
        comparison: unresolvedComparison('canonical_not_requested', 0),
      });
      return;
    }

    const controller = new AbortController();
    setComparisonState({
      loading: true,
      comparison: unresolvedComparison('canonical_read_in_progress', rowCount),
    });
    void fetchCanonicalComparison(
      dataApiUrl,
      runtimeResult,
      fetch,
      controller.signal,
    ).then((comparison) => {
      if (!controller.signal.aborted) {
        setComparisonState({ loading: false, comparison });
      }
    });

    return () => controller.abort();
  }, [dataApiUrl, rowCount, runtimeResult]);

  return (
    <section
      className="blf2-tab-panel blf2-tab-panel-active"
      role="tabpanel"
      aria-label={activeTab}
    >
      {activeTab === 'Results' && <ResultsView runtimeResult={runtimeResult} />}
      {activeTab === 'Validation' && (
        <ValidationView
          runtimeResult={runtimeResult}
          comparisonState={comparisonState}
        />
      )}
      {activeTab === 'Metadata' && (
        <MetadataView
          selectedSource={selectedSource}
          runtimeResult={runtimeResult}
        />
      )}
      {activeTab === 'Provenance' && (
        <ProvenanceView
          selectedSource={selectedSource}
          runtimeResult={runtimeResult}
          comparisonState={comparisonState}
        />
      )}
    </section>
  );
}
