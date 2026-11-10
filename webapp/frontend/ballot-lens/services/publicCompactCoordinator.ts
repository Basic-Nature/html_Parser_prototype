/** P6C5C3 offline integration candidate; activation / repository apply is gated. */
import { PublicCompactAssembler, type CompactCommand, type CompactSummary } from './publicCompactAssembler';

export type CompactWireEvent =
  | 'public_registry_compact_manifest'
  | 'public_registry_compact_page'
  | 'public_registry_compact_complete'
  | 'public_registry_compact_cancelled'
  | 'public_registry_compact_unavailable';

export const COMPACT_WIRE_EVENTS: readonly CompactWireEvent[] = Object.freeze([
  'public_registry_compact_manifest',
  'public_registry_compact_page',
  'public_registry_compact_complete',
  'public_registry_compact_cancelled',
  'public_registry_compact_unavailable',
]);

export interface PendingCompactRun {
  readonly sourceId: string;
  readonly rows: number;
  readonly pageCount: number;
}

export class PublicCompactCoordinator {
  private pending: PendingCompactRun | null = null;
  private assembler: PublicCompactAssembler | null = null;
  private hasManifest = false;

  get active(): boolean { return this.pending !== null; }
  get progress(): Readonly<{ pages: number; totalPages: number; rows: number; totalRows: number }> | null {
    return this.assembler?.progress ?? null;
  }
  constructor(
    private readonly emitCommand: (command: CompactCommand) => void,
    private readonly nowSeconds: () => number,
  ) {}

  start(pending: PendingCompactRun): void {
    if (this.active) throw new Error('Compact run already pending');
    if (!pending.sourceId || !Number.isSafeInteger(pending.rows) || pending.rows <= 0 ||
        !Number.isSafeInteger(pending.pageCount) || pending.pageCount <= 0 || pending.pageCount > pending.rows) {
      throw new Error('Invalid compact-ready envelope');
    }
    this.pending = Object.freeze({ ...pending });
    this.assembler = new PublicCompactAssembler();
    this.hasManifest = false;
  }

  discard(): void {
    this.pending = null;
    this.assembler = null;
    this.hasManifest = false;
  }

  handle(event: CompactWireEvent, payload: unknown, selectedSourceId: string): CompactSummary | null {
    const pending = this.pending;
    const assembler = this.assembler;
    if (!pending || !assembler || pending.sourceId !== selectedSourceId) {
      throw new Error('No owned pending compact run');
    }
    try {
      if (event === 'public_registry_compact_manifest') {
        if (this.hasManifest) throw new Error('Duplicate compact manifest');
        const command = assembler.begin(payload, selectedSourceId, this.nowSeconds());
        const progress = assembler.progress;
        if (progress.totalRows !== pending.rows || progress.totalPages !== pending.pageCount) {
          throw new Error('Compact manifest/result mismatch');
        }
        this.hasManifest = true;
        this.emitCommand(command);
        return null;
      }
      if (!this.hasManifest) throw new Error('Compact frame without manifest');
      if (event === 'public_registry_compact_page') {
        this.emitCommand(assembler.page(payload, this.nowSeconds()));
        return null;
      }
      if (event === 'public_registry_compact_complete') {
        const complete = assembler.complete(payload);
        this.discard();
        return complete;
      }
      if (event === 'public_registry_compact_cancelled') {
        assembler.cancelled(payload);
        this.discard();
        return null;
      }
      throw new Error('Compact delivery unavailable');
    } catch (error) {
      this.discard();
      throw error;
    }
  }
}
