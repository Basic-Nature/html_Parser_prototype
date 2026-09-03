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
import type { RunEvent, RunMode } from '../contracts/runtime';
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
  installTrustedRuntimeLifecycle,
  submitTrustedSource,
  type TrustedSourceSelection,
} from '../services/trustedExecution';
import {
  createInitialRunState,
  reduceRunState,
} from '../state/runMachine';
import {
  canSubmit,
  canSubmitApprovedRegistrySource,
} from '../state/selectors';

export function AppShell({
  bootstrap,
}: {
  readonly bootstrap: BallotLensBootstrap;
}) {
  const [registryEnvelope, setRegistryEnvelope] =
    useState<PublicRegistryEnvelope | null>(null);
  const [selectedSource, setSelectedSource] =
    useState<PublicRegistrySource | null>(null);
  const [activeMode, setActiveMode] =
    useState<RunMode>('public_registry');
  const [trustedSelection, setTrustedSelection] =
    useState<TrustedSourceSelection | null>(null);
  const [submitError, setSubmitError] =
    useState<string | null>(null);
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
  const trustedSelectionRef = useRef(trustedSelection);

  const dispatch = useCallback((event: RunEvent) => {
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
      dispatch({ type: 'RESET' });
    }
  }, [dispatch]);

  const handleModeChange = useCallback((mode: RunMode) => {
    if (
      selectionLocked
      || mode === activeMode
      || (mode !== 'public_registry' && !bootstrap.trustedControls)
    ) {
      return;
    }
    setActiveMode(mode);
    setSubmitError(null);
    setPublicRuntimeResult(null);
    setSelectedSource(null);
    selectedSourceRef.current = null;
    setTrustedSelection(null);
    trustedSelectionRef.current = null;
    dispatch({ type: 'RESET' });
  }, [
    activeMode,
    bootstrap.trustedControls,
    dispatch,
    selectionLocked,
  ]);

  const handlePublicSelection = useCallback((
    source: PublicRegistrySource | null,
  ) => {
    setSubmitError(null);
    setPublicRuntimeResult(null);
    setSelectedSource(source);
    selectedSourceRef.current = source;
    if (!source) {
      dispatch({ type: 'RESET' });
      return;
    }
    dispatch({
      type: 'SOURCE_SELECTED',
      runMode: 'public_registry',
      sourceSummary: {
        runMode: 'public_registry',
        displayLabel: registrySourceLabel(source),
        registrySourceId: source.registry_source_id,
      },
    });
  }, [dispatch]);

  const handleTrustedSelection = useCallback((
    selection: TrustedSourceSelection | null,
  ) => {
    setSubmitError(null);
    setPublicRuntimeResult(null);
    setTrustedSelection(selection);
    trustedSelectionRef.current = selection;
    if (!selection) {
      dispatch({ type: 'RESET' });
      return;
    }
    dispatch({
      type: 'SOURCE_SELECTED',
      runMode: selection.runMode,
      sourceSummary: {
        runMode: selection.runMode,
        displayLabel: selection.displayLabel,
      },
    });
  }, [dispatch]);

  const runEligible = (
    activeMode === 'public_registry'
    && canSubmitApprovedRegistrySource(registryEnvelope, selectedSource)
    && canSubmit(runState)
  ) || (
    activeMode !== 'public_registry'
    && bootstrap.trustedControls
    && trustedSelection?.runMode === activeMode
    && runState.context.runMode === activeMode
    && canSubmit(runState)
  );

  const handleRun = useCallback(() => {
    if (!canSubmit(runState)) {
      return;
    }
    setSubmitError(null);
    setPublicRuntimeResult(null);
    dispatch({ type: 'SUBMIT_REQUESTED' });
    try {
      if (activeMode === 'public_registry') {
        if (
          !selectedSource
          || !canSubmitApprovedRegistrySource(
            registryEnvelope,
            selectedSource,
          )
        ) {
          dispatch({ type: 'RESET' });
          return;
        }
        submitApprovedRegistrySource(
          socket,
          selectedSource.registry_source_id,
        );
      } else {
        if (
          !bootstrap.trustedControls
          || !trustedSelection
          || trustedSelection.runMode !== activeMode
        ) {
          dispatch({ type: 'RESET' });
          return;
        }
        submitTrustedSource(socket, trustedSelection);
      }
      dispatch({ type: 'SUBMISSION_ACCEPTED' });
    } catch (error: unknown) {
      dispatch({ type: 'RESET' });
      setSubmitError(
        error instanceof Error
          ? error.message
          : 'Selected source submission could not be dispatched.',
      );
    }
  }, [
    activeMode,
    bootstrap.trustedControls,
    dispatch,
    registryEnvelope,
    runState,
    selectedSource,
    socket,
    trustedSelection,
  ]);

  useEffect(() => {
    const detachLifecycle = installPublicRuntimeLifecycle(socket, {
      getRunState: () => runStateRef.current,
      getSelectedRegistrySourceId: () => (
        selectedSourceRef.current?.registry_source_id ?? null
      ),
      dispatch,
      onRuntimeResult: setPublicRuntimeResult,
      onProtocolError: () => setSubmitError(
        'Invalid public runtime lifecycle result.',
      ),
    });
    const detachTrustedLifecycle = installTrustedRuntimeLifecycle(socket, {
      getRunState: () => runStateRef.current,
      getSelection: () => trustedSelectionRef.current,
      dispatch,
      onProtocolError: () => setSubmitError(
        'Invalid trusted runtime lifecycle evidence.',
      ),
    });

    return () => {
      detachLifecycle();
      detachTrustedLifecycle();
      if (socket.connected) socket.disconnect();
    };
  }, [dispatch, socket]);

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
          uploadedFiles={bootstrap.uploadedFiles}
          activeMode={activeMode}
          selectedSourceId={selectedSource?.registry_source_id ?? ''}
          trustedSelection={trustedSelection}
          selectionLocked={selectionLocked}
          onModeChange={handleModeChange}
          onRegistryEnvelopeChange={handleRegistryEnvelopeChange}
          onSelectionChange={handlePublicSelection}
          onTrustedSelectionChange={handleTrustedSelection}
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
        <CheckpointRail runState={runState} />
      </main>
      <DiagnosticsDrawer runState={runState} />
    </div>
  );
}
