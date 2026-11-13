/** B12 component unit tests: synthetic counters, never published precinct previews. */
import { describe, expect, it } from 'vitest';
import { paginatePreviewRows, PUBLIC_RESULT_PAGE_SIZE } from '../components/workspace/WorkspaceViews';

describe('B12 workspace viewport pagination', () => {
  it('never truncates source rows while showing only one bounded page', () => {
    const source = Array.from({ length: 1959 }, (_, index) => index);
    expect(PUBLIC_RESULT_PAGE_SIZE).toBe(25);
    expect(paginatePreviewRows(source, 0)).toEqual(source.slice(0, 25));
    expect(paginatePreviewRows(source, 1)).toEqual(source.slice(25, 50));
    expect(paginatePreviewRows(source, 78)).toEqual(source.slice(1950, 1959));
    expect(source).toHaveLength(1959);
  });
  it('rejects invalid page coordinates and does not repeat rows', () => {
    const source = [0, 1, 2, 3, 4];
    expect(paginatePreviewRows(source, -1)).toEqual([]);
    expect(paginatePreviewRows(source, Number.NaN)).toEqual([]);
    expect(paginatePreviewRows(source, 0, 2)).toEqual([0, 1]);
    expect(paginatePreviewRows(source, 1, 2)).toEqual([2, 3]);
    expect(paginatePreviewRows(source, 2, 2)).toEqual([4]);
  });
});
