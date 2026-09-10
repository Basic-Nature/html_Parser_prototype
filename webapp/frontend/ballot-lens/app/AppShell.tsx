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
  clearWorkflowHandoffQuery,
  readWorkflowHandoffQueryIntent,
  type WorkflowHandoffQueryIntent,
} from '../services/workflowHandoff';
import {
  createInitialRunState,
  reduceRunState,
} from '../state/runMachine';
import {
  EMPTY_SESSION_HISTORY,
  captureOwnedSession,
  type SessionHistory,
} from '../state/sessionHistory';
import {
  canSubmit,
  canSubmitApprovedRegistrySource,
} from '../state/selectors';

const SHAREABLE_SOURCE_QUERY_KEY = 'source';
const PUBLIC_REGISTRY_SOURCE_ID_INTENT_RX =
  /^blsrc_v1_[0-9a-f]{64}$/;

interface SourceQueryIntent {
  readonly present: boolean;
  readonly registrySourceId: string | null;
}

function normalizeSourceQueryIntent(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const normalized = value.trim();
  return PUBLIC_REGISTRY_SOURCE_ID_INTENT_RX.test(normalized)
    ? normalized
    : null;
}

function readSourceQueryIntent(): SourceQueryIntent {
  if (typeof window === 'undefined') {
    return { present: false, registrySourceId: null };
  }
  const params = new URLSearchParams(window.location.search);
  return {
    present: params.has(SHAREABLE_SOURCE_QUERY_KEY),
    registrySourceId: normalizeSourceQueryIntent(
      params.get(SHAREABLE_SOURCE_QUERY_KEY),
    ),
  };
}

function replaceSourceQueryIntent(registrySourceId: string | null): void {
  if (
    typeof window === 'undefined'
    || !window.history?.replaceState
  ) {
    return;
  }
  const normalized = normalizeSourceQueryIntent(registrySourceId);
  const url = new URL(window.location.href);
  if (normalized) {
    url.searchParams.set(SHAREABLE_SOURCE_QUERY_KEY, normalized);
  } else {
    url.searchParams.delete(SHAREABLE_SOURCE_QUERY_KEY);
  }
  const nextLocation = `${url.pathname}${url.search}${url.hash}`;
  const currentLocation =
    `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (nextLocation !== currentLocation) {
    window.history.replaceState(window.history.state, '', nextLocation);
  }
}

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
  const [sessionHistory, setSessionHistory] =
    useState<SessionHistory>(EMPTY_SESSION_HISTORY);
  const [diagnosticSessionId, setDiagnosticSessionId] =
    useState<string | null>(null);
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
  const sourceQueryHydratedRef = useRef(false);
  const initialSourceQueryIntentRef = useRef<SourceQueryIntent | null>(null);
  const initialWorkflowHandoffIntentRef =
    useRef<WorkflowHandoffQueryIntent | null>(null);
  if (initialSourceQueryIntentRef.current === null) {
    initialSourceQueryIntentRef.current = readSourceQueryIntent();
  }
  if (initialWorkflowHandoffIntentRef.current === null) {
    initialWorkflowHandoffIntentRef.current =
      readWorkflowHandoffQueryIntent();
  }

  const dispatch = useCallback((event: RunEvent) => {
    const nextState = reduceRunState(runStateRef.current, event);
    runStateRef.current = nextState;

    if (nextState.context.sessionId) {
      setSessionHistory((current) => (
        captureOwnedSession(current, nextState, event)
      ));
      setDiagnosticSessionId((current) => (
        event.type === 'SESSION_CORRELATED'
          ? nextState.context.sessionId
          : current ?? nextState.context.sessionId
      ));
    }

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
    replaceSourceQueryIntent(null);
    clearWorkflowHandoffQuery();
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
    clearWorkflowHandoffQuery();
    setSubmitError(null);
    setPublicRuntimeResult(null);
    setSelectedSource(source);
    selectedSourceRef.current = source;
    replaceSourceQueryIntent(source?.registry_source_id ?? null);
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

  useEffect(() => {
    if (
      !registryEnvelope
      || sourceQueryHydratedRef.current
    ) {
      return;
    }
    sourceQueryHydratedRef.current = true;

    const intent = initialSourceQueryIntentRef.current;
    if (!intent?.present) {
      return;
    }
    if (!intent.registrySourceId) {
      replaceSourceQueryIntent(null);
      return;
    }

    // Query state is locator intent only. Server-projected registry sources
    // remain the sole selection authority.
    const matches = registryEnvelope.sources.filter(
      source => source.registry_source_id === intent.registrySourceId,
    );
    if (matches.length !== 1) {
      replaceSourceQueryIntent(null);
      return;
    }
    handlePublicSelection(matches[0] ?? null);
  }, [handlePublicSelection, registryEnvelope]);

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

  useEffect(() => {
    const intent = initialWorkflowHandoffIntentRef.current;
    if (!intent?.present) {
      return;
    }

    if (
      initialSourceQueryIntentRef.current?.present
      || !bootstrap.trustedControls
      || !intent.selection
    ) {
      clearWorkflowHandoffQuery();
      return;
    }

    setActiveMode('worklist');
    setSelectedSource(null);
    selectedSourceRef.current = null;
    replaceSourceQueryIntent(null);
    handleTrustedSelection(intent.selection);
  }, [
    bootstrap.trustedControls,
    handleTrustedSelection,
  ]);

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
      <DiagnosticsDrawer
        runState={runState}
        sessionHistory={sessionHistory}
        selectedSessionId={diagnosticSessionId}
        onSelectSession={setDiagnosticSessionId}
      />
    </div>
  );
}
