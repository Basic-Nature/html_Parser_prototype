import {
  io,
  type ManagerOptions,
  type SocketOptions,
} from 'socket.io-client';

import type {
  SocketIoClientBootstrap,
} from '../contracts/bootstrap';
import type {
  PublicRegistrySubmitPayload,
} from './publicSubmit';
import type {
  TrustedBallotLensSubmitPayload,
} from './trustedExecution';

export interface DormantBallotLensSocket {
  readonly connected: boolean;
  connect(): DormantBallotLensSocket;
  disconnect(): DormantBallotLensSocket;
  on(
    event: string,
    listener: (payload?: unknown) => void,
  ): DormantBallotLensSocket;
  off(
    event: string,
    listener?: (payload?: unknown) => void,
  ): DormantBallotLensSocket;
  emit(
    event: 'ballot_lens',
    payload: PublicRegistrySubmitPayload | TrustedBallotLensSubmitPayload,
  ): DormantBallotLensSocket;
}

export interface DormantSocketFactoryOptions {
  readonly autoConnect: false;
  readonly transports: readonly ('websocket' | 'polling')[];
  readonly upgrade: boolean;
}

export type DormantSocketFactory = (
  options: DormantSocketFactoryOptions,
) => DormantBallotLensSocket;

function defaultSocketFactory(
  options: DormantSocketFactoryOptions,
): DormantBallotLensSocket {
  const socketOptions = {
    autoConnect: options.autoConnect,
    transports: [...options.transports],
    upgrade: options.upgrade,
  } as unknown as Partial<ManagerOptions & SocketOptions>;

  return io(socketOptions) as unknown as DormantBallotLensSocket;
}

export function createDormantBallotLensSocket(
  config: SocketIoClientBootstrap,
  factory: DormantSocketFactory = defaultSocketFactory,
): DormantBallotLensSocket {
  return factory(Object.freeze({
    autoConnect: false as const,
    transports: Object.freeze([...config.transports]),
    upgrade: config.upgrade,
  }));
}
