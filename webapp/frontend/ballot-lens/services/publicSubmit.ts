import type {
  DormantBallotLensSocket,
} from './socketClient';

export const PUBLIC_BALLOT_LENS_COMMAND_EVENT = 'ballot_lens' as const;

export interface PublicRegistrySubmitPayload {
  readonly registry_source_id: string;
}

export function submitApprovedRegistrySource(
  socket: DormantBallotLensSocket,
  registrySourceId: string,
): void {
  const normalizedSourceId = registrySourceId.trim();
  if (!normalizedSourceId) {
    throw new Error('Approved registry source id is required');
  }

  socket.connect();
  socket.emit(
    PUBLIC_BALLOT_LENS_COMMAND_EVENT,
    Object.freeze({ registry_source_id: normalizedSourceId }),
  );
}
