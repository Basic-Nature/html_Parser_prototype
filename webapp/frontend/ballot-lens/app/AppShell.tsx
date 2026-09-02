import {
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from 'react';

import type { BallotLensBootstrap } from '../contracts/bootstrap';
import {
  registrySourceLabel,
  type PublicRegistryEnvelope,
  type PublicRegistrySource,
} from '../contracts/registry';
import type { PublicRuntimeResult } from '../contracts/publicRuntime';
import type { RunEvent } from '../contracts/runtime';
import { HeaderBar } from '../components/common/HeaderBar';
import { CheckpointRail } from '../components/checkpoints/CheckpointRail';
import { DiagnosticsDrawer } from '../components/diagnostics/DiagnosticsDrawer';
import { SourcePanel } from '../components/source/SourcePanel';
import { CosmicBackdrop } from '../components/theme/CosmicBackdrop';
import { WorkspaceShell } from '../components/workspace/WorkspaceShell';
import { submitApprovedRegistrySource } from '../services/publicSubmit';
import { installPublicRuntimeLifecycle } from '../services/publicRuntimeLifecycle';
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
  const [publicRuntimeResult, setPublicRuntimeResult] =
    useState<PublicRuntimeResult | null>(null);
  const [runState, dispatchRunEvent] = useReducer(
    reduceRunState,
    undefined,
    () => createInitialRunState(),
  );
  const socket = useMemo(
    () => createDormantBallotLensSocket(bootstrap.socketIo),
    [bootstrap.socketIo],
  );
  const runStateRef = useRef(runState);
  const selectedSourceRef = useRef(selectedSource);

  const dispatchOwnedRunEvent = useCallback((event: RunEvent) => {
    runStateRef.current = reduceRunState(runStateRef.current, event);
    dispatchRunEvent(event);
  }, []);

  const selectionLocked = ![
    'idle',
    'source_selected',
    'terminal',
  ].includes(runState.status);

  const handleRegistryEnvelopeChange = useCallback((
    envelope: PublicRegistryEnvelope | null,
  ) => {
    setRegistryEnvelope(envelope);
    setPublicRuntimeResult(null);
    if (!envelope) {
      setSelectedSource(null);
      selectedSourceRef.current = null;
      dispatchOwnedRunEvent({ type: 'RESET' });
    }
  }, [dispatchOwnedRunEvent]);

  const handleSelectionChange = useCallback((
    source: PublicRegistrySource | null,
  ) => {
    setSubmitError(null);
    setPublicRuntimeResult(null);
    setSelectedSource(source);
    selectedSourceRef.current = source;
    if (!source) {
      dispatchOwnedRunEvent({ type: 'RESET' });
      return;
    }
    dispatchOwnedRunEvent({
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: {
        runMode: 'public_registry',
        displayLabel: registrySourceLabel(source),
        registrySourceId: source.registry_source_id,
      },
    });
  }, [dispatchOwnedRunEvent]);

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
    setPublicRuntimeResult(null);
    dispatchOwnedRunEvent({ type: 'SUBMIT_REQUESTED' });
    try {
      submitApprovedRegistrySource(
        socket,
        selectedSource.registry_source_id,
      );
      dispatchOwnedRunEvent({ type: 'SUBMISSION_ACCEPTED' });
    } catch (error: unknown) {
      dispatchOwnedRunEvent({ type: 'RESET' });
      dispatchOwnedRunEvent({
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
  }, [
    dispatchOwnedRunEvent,
    registryEnvelope,
    runState,
    selectedSource,
    socket,
  ]);

  useEffect(() => {
    const detachLifecycle = installPublicRuntimeLifecycle(socket, {
      getRunState: () => runStateRef.current,
      getSelectedRegistrySourceId: () => (
        selectedSourceRef.current?.registry_source_id ?? null
      ),
      dispatch: dispatchOwnedRunEvent,
      onRuntimeResult: setPublicRuntimeResult,
      onProtocolError: () => setSubmitError(
        'The public runtime returned an invalid lifecycle result.',
      ),
    });

    return () => {
      detachLifecycle();
      if (socket.connected) socket.disconnect();
    };
  }, [dispatchOwnedRunEvent, socket]);

  return (
    <div
      className="blf2-app"
      data-phase={bootstrap.phase}
      data-runtime-result-ready={publicRuntimeResult !== null}
    >
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
          runtimeResult={publicRuntimeResult}
          dataApiUrl={bootstrap.dataApiUrl}
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
