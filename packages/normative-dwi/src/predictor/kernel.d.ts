import type { Covariates, MarkerParams, ZmapResult } from "../model.js";

/**
 * Evaluate the 4 nonzero cubic B-spline basis functions at x into `out` and
 * return the index of the first one (end pieces extended outside the base interval).
 */
export function bsplineBasis(knots: Float64Array, offset: number, nKnots: number, x: number, out: Float64Array, left: Float64Array, right: Float64Array): number;
/** Cholesky factor of one voxel's precision A (upper triangle at aTriu[offset…]) into the lower triangle of L (D×D, row-major). */
export function choleskyLower(aTriu: ArrayLike<number>, offset: number, D: number, L: Float64Array): void;
/** Index of the subject's sex in be_map columns; throws if the store lacks it. */
export function sexColumn(params: Pick<MarkerParams, "beLevelColumns">, covariates: Covariates): number;
/** Voxels scored between checks for cancellation in zmapCpuAsync. */
export const BLOCK: number;
/** Z-scores of `y` (one value per store voxel) for one subject. */
export function zmapCpu(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>): ZmapResult;
/** zmapCpu in blocks, yielding to the event loop between them so `signal` can cancel. */
export function zmapCpuAsync(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal, block?: number): Promise<ZmapResult>;
