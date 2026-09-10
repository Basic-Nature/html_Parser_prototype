import { useEffect, useState } from 'react';

import type { RunMode } from '../../contracts/runtime';
import {
  fetchTrustedUrlLibrary,
  type TrustedChoice,
  type TrustedSourceSelection,
} from '../../services/trustedExecution';

export function TrustedSourceBrowser({
  activeMode,
  uploadedFiles,
  selectionLocked,
  selected,
  onSelectionChange,
}: {
  activeMode: Exclude<RunMode, 'public_registry'>;
  uploadedFiles: readonly string[];
  selectionLocked: boolean;
  selected: TrustedSourceSelection | null;
  onSelectionChange: (selection: TrustedSourceSelection | null) => void;
}) {
  const [choices, setChoices] = useState<readonly TrustedChoice[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setChoices([]);
    setError(null);

    if (activeMode !== 'trusted_url') {
      return () => {
        cancelled = true;
      };
    }

    fetchTrustedUrlLibrary()
      .then(next => {
        if (!cancelled) setChoices(next);
      })
      .catch(() => {
        if (!cancelled) setError('Trusted source list unavailable.');
      });

    return () => {
      cancelled = true;
    };
  }, [activeMode]);

  if (activeMode === 'manual_upload') {
    const value =
      selected?.runMode === 'manual_upload' ? selected.uploadPath : '';
    return (
      <section className="blf2-registry" aria-label="Trusted uploads">
        <form
          method="post"
          action="/ballot_lens"
          encType="multipart/form-data"
        >
          <label>
            Upload a new artifact
            <input
              type="file"
              name="data_file"
              disabled={selectionLocked}
            />
          </label>
          <button type="submit" disabled={selectionLocked}>
            Upload with existing trusted gate
          </button>
        </form>
        <label>
          Select an uploaded file
          <select
            value={value}
            disabled={selectionLocked}
            onChange={event => {
              const uploadPath = event.currentTarget.value;
              if (!uploadPath) {
                onSelectionChange(null);
                return;
              }
              const uploadName =
                uploadPath.split('/').pop() ?? uploadPath;
              onSelectionChange(Object.freeze({
                runMode: 'manual_upload' as const,
                displayLabel: uploadName,
                uploadPath,
                uploadName,
              }));
            }}
          >
            <option value="">Choose existing upload</option>
            {uploadedFiles.map(path => (
              <option key={path} value={path}>{path}</option>
            ))}
          </select>
        </label>
      </section>
    );
  }

  if (activeMode === 'worklist') {
    const handoff =
      selected?.runMode === 'worklist' ? selected : null;
    return (
      <section className="blf2-registry" aria-label="Governed Worklist">
        <h3>Governed Worklist</h3>
        <span className="blf2-panel-state">
          Workflow authority handoff
        </span>
        <p>
          {handoff
            ? 'Governed Workflow task ready. Assignment, row version, source trust, and execution capability are revalidated by the server when Run is pressed.'
            : 'Choose an assigned acquisition task from Workflow to establish a governed Ballot Lens handoff.'}
        </p>
        <a href="/worklist">Open Workflow</a>
      </section>
    );
  }

  const value =
    selected?.runMode === 'trusted_url' ? selected.url : '';
  return (
    <section
      className="blf2-registry"
      aria-label="Approved trusted URL Library"
    >
      <h3>Approved URL Library</h3>
      <span className="blf2-panel-state">
        Backend registry gate remains authoritative
      </span>
      {error && <p role="alert">{error}</p>}
      <label>
        Select an existing reviewed target
        <select
          value={value}
          disabled={selectionLocked}
          onChange={event => {
            const url = event.currentTarget.value;
            const choice = choices.find(item => item.url === url);
            if (!choice) {
              onSelectionChange(null);
              return;
            }
            onSelectionChange(Object.freeze({
              runMode: 'trusted_url' as const,
              displayLabel: choice.displayLabel,
              url: choice.url,
            }));
          }}
        >
          <option value="">Choose reviewed target</option>
          {choices.map(choice => (
            <option key={choice.url} value={choice.url}>
              {choice.displayLabel}
            </option>
          ))}
        </select>
      </label>
      <p>
        Selection is not authorization; backend reviewed-registry and
        guarded-ingestion checks remain final.
      </p>
    </section>
  );
}
