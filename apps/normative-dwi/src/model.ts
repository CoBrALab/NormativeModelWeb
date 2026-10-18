// Domain types shared by the store reader, the predictors and the image I/O.
// Vocabulary: see docs/GLOSSARY.md.

export const MARKERS = ["FA", "MD", "ICVF", "ISOVF", "OD"] as const;
export type Marker = (typeof MARKERS)[number];

export type Sex = "female" | "male";

/** The subject's covariates as entered; age in years. */
export interface Covariates {
  age: number;
  sex: Sex;
}

/** Store batch-effect level for a sex ("Sex=0" is female in the UK Biobank models). */
export function sexLevel(sex: Sex): string {
  return sex === "female" ? "Sex=0" : "Sex=1";
}

/** Template space: the grid every map and voxel model lives on. */
export interface Grid {
  /** Array shape in store (C) order. */
  shape: readonly [number, number, number];
  /** Store voxel index (i, j, k in store order) → world mm, row-major 4×4. */
  affine: Float64Array;
  /** Store voxel index of each modelled-grid row, nVox × 3 (store order). */
  ijk: Int32Array;
  nVox: number;
}

/**
 * One marker's voxel models, struct-of-arrays, row j = store voxel j.
 * Layout and meaning follow nmzarr format 0.1 (proc_nm/opt_inference docs/reference/format.md).
 */
export interface MarkerParams {
  marker: Marker;
  nVox: number;
  /** Mean B-spline knots per voxel (cubic, so nKnots − 4 basis functions). */
  nKnots: number;
  /** Batch-effect level names in be_map column order, e.g. ["Sex=0", "Sex=1"]. */
  beLevelColumns: readonly string[];
  hasModel: Uint8Array;
  /** Posterior mean weights, nVox × D. */
  m: Float64Array;
  /** Posterior precision, upper triangle row-major, nVox × D(D+1)/2. */
  aTriu: Float64Array;
  /** [w_var0, w_var1, epsilon, log b, log alpha], nVox × 5. */
  hyp: Float64Array;
  meanKnots: Float64Array;
  /** Age scaler [mean, sd, min, max], nVox × 4. */
  inscaler: Float64Array;
  /** Response scaler [mean, sd, min, max], nVox × 4. */
  outscaler: Float64Array;
  /** Design-matrix one-hot column of each batch-effect level, nVox × nBeLevels. */
  beMap: Int16Array;
  /** Training range of age [min, max], nVox × 2. */
  covRanges: Float64Array;
}

export interface ZmapResult {
  /** Z-score per store voxel; 0 where not valid. */
  z: Float64Array;
  /** 1 where the voxel is modelled and the map has a value. */
  valid: Uint8Array;
  /** Number of valid voxels whose age was outside the training range. */
  clamped: number;
}

/** Store design: the D = basis + intercept + one-hot batch columns. */
export function designSize(params: Pick<MarkerParams, "nKnots" | "beLevelColumns">): number {
  return params.nKnots - 4 + 1 + params.beLevelColumns.length;
}
