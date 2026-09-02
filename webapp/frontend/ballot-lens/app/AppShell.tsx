import {
  useCallback,
  useMemo,
  useReducer,
  useState,
} from 'react';

import type { BallotLensBootstrap } from '../contracts/bootstrap';
import {
  registrySourceLabel,
  type PublicRegistryEnvelope,
  type PublicRegistrySource,
} from '../contracts/registry';
import { HeaderBar } from '../components/common/HeaderBar';
import { CheckpointRail } from '../components/checkpoints/CheckpointRail';
import { DiagnosticsDrawer } from '../components/diagnostics/DiagnosticsDrawer';
import { SourcePanel } from '../components/source/SourcePanel';
import { CosmicBackdrop } from '../components/theme/CosmicBackdrop';
import { WorkspaceShell } from '../components/workspace/WorkspaceShell';
import { submitApprovedRegistrySource } from '../services/publicSubmit';
import { createDormantBallotLensSocket } from '../services/socketClient';
import {
  createInitialRunState,
  reduceRunState,
} from '../state/runMachine';
import {
  canSubmit,
  canSubmitApprovedRegistrySource,
} from '../state/selectors';

interface AppShellProps {
  readonly bootstrap: BallotLensBootstrap;
}

export function AppShell({ bootstrap }: AppShellProps) {
  const [registryEnvelope, setRegistryEnvelope] =
    useState<PublicRegistryEnvelope | null>(null);
  const [selectedSource, setSelectedSource] =
    useState<PublicRegistrySource | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [runState, dispatchRunEvent] = useReducer(
    reduceRunState,
    undefined,
    () => createInitialRunState(),
  );
  const socket = useMemo(
    () => createDormantBallotLensSocket(bootstrap.socketIo),
    [bootstrap.socketIo],
  );

  const selectionLocked = ![
    'idle',
    'source_selected',
  ].includes(runState.status);

  const handleRegistryEnvelopeChange = useCallback((
    envelope: PublicRegistryEnvelope | null,
  ) => {
    setRegistryEnvelope(envelope);
    if (!envelope) {
      setSelectedSource(null);
      dispatchRunEvent({ type: 'RESET' });
    }
  }, []);

  const handleSelectionChange = useCallback((
    source: PublicRegistrySource | null,
  ) => {
    setSubmitError(null);
    setSelectedSource(source);
    if (!source) {
      dispatchRunEvent({ type: 'RESET' });
      return;
    }
    dispatchRunEvent({
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: {
        runMode: 'public_registry',
        displayLabel: registrySourceLabel(source),
        registrySourceId: source.registry_source_id,
      },
    });
  }, []);

  const runEligible = canSubmitApprovedRegistrySource(
    registryEnvelope,
    selectedSource,
  ) && canSubmit(runState);

  const handleRun = useCallback(() => {
    if (
      !selectedSource
      || !canSubmitApprovedRegistrySource(registryEnvelope, selectedSource)
      || !canSubmit(runState)
    ) {
      return;
    }

    setSubmitError(null);
    dispatchRunEvent({ type: 'SUBMIT_REQUESTED' });
    try {
      submitApprovedRegistrySource(
        socket,
        selectedSource.registry_source_id,
      );
      dispatchRunEvent({ type: 'SUBMISSION_ACCEPTED' });
    } catch (error: unknown) {
      dispatchRunEvent({ type: 'RESET' });
      dispatchRunEvent({
        type: 'SOURCE_SELECTED',
        runMode: 'public_registry',
        sourceSummary: {
          runMode: 'public_registry',
          displayLabel: registrySourceLabel(selectedSource),
          registrySourceId: selectedSource.registry_source_id,
        },
      });
      setSubmitError(
        error instanceof Error
          ? error.message
          : 'Approved source submission could not be dispatched.',
      );
    }
  }, [registryEnvelope, runState, selectedSource, socket]);

  return (
    <div className="blf2-app" data-phase={bootstrap.phase}>
      <CosmicBackdrop />
      <HeaderBar bootstrap={bootstrap} runState={runState} />
      <main className="blf2-shell">
        <SourcePanel
          trustedControls={bootstrap.trustedControls}
          publicRegistryApi={bootstrap.publicRegistryApi}
          selectedSourceId={selectedSource?.registry_source_id ?? ''}
          selectionLocked={selectionLocked}
          onRegistryEnvelopeChange={handleRegistryEnvelopeChange}
          onSelectionChange={handleSelectionChange}
        />
        <WorkspaceShell
          selectedSource={selectedSource}
          runState={runState}
          canRun={runEligible}
          submitError={submitError}
          onRun={handleRun}
        />
        <CheckpointRail />
      </main>
      <DiagnosticsDrawer runState={runState} />
    </div>
  );
}
