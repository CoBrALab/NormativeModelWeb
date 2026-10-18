// Float64 z-score kernel: the reference implementation (docs/adr/0001).
//
// Reproduces PCNtoolkit 1.4.0 BLR prediction (fork gdevenyi/PCNToolkit@4219457,
// not released) for the nmzarr "ukb_blr_agesex"
// family (warped sinh-arcsinh, reparametrised; heteroskedastic noise on
// [x, 1]; cubic B-spline mean basis; one-hot batch-effect intercepts):
//
//   x   = (age − in_m) / in_s                        age clamped to the training range
//                                                    (proc_nm's convention, not native PCNtoolkit)
//   Φ   = [B(x), 1, onehot(be_map[sex])]             Φv = [x, 1]
//   b   = exp(hyp3),  a = −hyp2·b,  β = exp(Φv·hyp[0:2]) / b²
//   μ   = Φ·m,        s² = 1/β + Φ A⁻¹ Φᵀ            A = posterior precision
//   z   = (sinh(b·asinh((y − out_m)/out_s) − a) − μ) / √s²
//
// The WebGPU adapter (webgpu.ts) follows the same steps in the same order.
import { designSize, sexLevel, type Covariates, type MarkerParams, type ZmapResult } from "../model.ts";

const DEGREE = 3;

/**
 * Evaluate the DEGREE+1 nonzero cubic B-spline basis functions at x into `out`
 * and return the index of the first one. Outside the base interval the end
 * polynomial pieces are extended (scipy BSpline extrapolate=True).
 */
export function bsplineBasis(knots: Float64Array, offset: number, nKnots: number, x: number, out: Float64Array, left: Float64Array, right: Float64Array): number {
  const nBasis = nKnots - DEGREE - 1;
  let i = DEGREE;
  while (i < nBasis - 1 && knots[offset + i + 1] <= x) i++;
  out[0] = 1;
  for (let j = 1; j <= DEGREE; j++) {
    left[j] = x - knots[offset + i + 1 - j];
    right[j] = knots[offset + i + j] - x;
    let saved = 0;
    for (let r = 0; r < j; r++) {
      const temp = out[r] / (right[r + 1] + left[j - r]);
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
  return i - DEGREE;
}

/**
 * Cholesky factor of one voxel's precision A (upper triangle at aTriu[offset…],
 * row-major) into the lower triangle of the D×D row-major array L.
 */
export function choleskyLower(aTriu: ArrayLike<number>, offset: number, D: number, L: Float64Array): void {
  for (let r = 0, t = offset; r < D; r++) {
    for (let c = r; c < D; c++, t++) L[c * D + r] = aTriu[t];
  }
  for (let c = 0; c < D; c++) {
    let diag = L[c * D + c];
    for (let k = 0; k < c; k++) diag -= L[c * D + k] * L[c * D + k];
    diag = Math.sqrt(diag);
    L[c * D + c] = diag;
    for (let r = c + 1; r < D; r++) {
      let s = L[r * D + c];
      for (let k = 0; k < c; k++) s -= L[r * D + k] * L[c * D + k];
      L[r * D + c] = s / diag;
    }
  }
}

/** Index of the subject's sex in be_map columns; throws if the store lacks it. */
export function sexColumn(params: Pick<MarkerParams, "beLevelColumns">, covariates: Covariates): number {
  const column = params.beLevelColumns.indexOf(sexLevel(covariates.sex));
  if (column < 0) throw new Error(`Store has no batch-effect level ${sexLevel(covariates.sex)}`);
  return column;
}

/** Voxels scored between checks for cancellation in zmapCpuAsync (~5 ms of work). */
export const BLOCK = 16384;

/**
 * Z-scores of `y` (one value per store voxel) for one subject.
 * A voxel is valid when it is modelled, y is finite and nonzero (0 = missing),
 * and its z-score is finite (a non-positive-definite A or variance gives none).
 */
export function zmapCpu(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>): ZmapResult {
  const scorer = createScorer(params, covariates, y);
  scorer.score(0, params.nVox);
  return scorer.result();
}

/** zmapCpu in blocks, yielding to the event loop between them so `signal` can cancel. */
export async function zmapCpuAsync(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal, block = BLOCK): Promise<ZmapResult> {
  const scorer = createScorer(params, covariates, y);
  for (let start = 0; start < params.nVox; start += block) {
    signal?.throwIfAborted();
    scorer.score(start, Math.min(start + block, params.nVox));
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  signal?.throwIfAborted();
  return scorer.result();
}

function createScorer(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>) {
  const { nVox, nKnots } = params;
  if (y.length !== nVox) throw new Error(`Expected ${nVox} values, got ${y.length}`);
  const D = designSize(params);
  const nTri = (D * (D + 1)) / 2;
  const nBe = params.beLevelColumns.length;
  const nBasis = nKnots - DEGREE - 1;
  const column = sexColumn(params, covariates);

  const z = new Float64Array(nVox);
  const valid = new Uint8Array(nVox);
  const phi = new Float64Array(D);
  const basis = new Float64Array(DEGREE + 1);
  const left = new Float64Array(DEGREE + 1);
  const right = new Float64Array(DEGREE + 1);
  const L = new Float64Array(D * D);
  let clamped = 0;

  function score(from: number, to: number) {
    for (let v = from; v < to; v++) {
      const value = y[v];
      if (!params.hasModel[v] || value === 0 || !Number.isFinite(value)) continue;

      // proc_nm convention: ages outside the training range are scored at its
      // nearest end. Native PCNtoolkit would extrapolate the B-spline instead.
      const lo = params.covRanges[2 * v];
      const hi = params.covRanges[2 * v + 1];
      let age = covariates.age;
      if (age < lo) age = lo;
      else if (age > hi) age = hi;
      const x = (age - params.inscaler[4 * v]) / params.inscaler[4 * v + 1];

      // Design row Φ.
      phi.fill(0);
      const first = bsplineBasis(params.meanKnots, v * nKnots, nKnots, x, basis, left, right);
      for (let r = 0; r <= DEGREE; r++) phi[first + r] = basis[r];
      phi[nBasis] = 1;
      phi[nBasis + 1 + params.beMap[v * nBe + column]] = 1;

      // μ = Φ·m
      let mu = 0;
      for (let d = 0; d < D; d++) mu += phi[d] * params.m[v * D + d];

      // Φ A⁻¹ Φᵀ via Cholesky A = L Lᵀ and forward substitution L w = Φ.
      choleskyLower(params.aTriu, v * nTri, D, L);
      let quad = 0;
      for (let r = 0; r < D; r++) {
        let s = phi[r];
        for (let k = 0; k < r; k++) s -= L[r * D + k] * phi[k];
        const w = s / L[r * D + r];
        phi[r] = w; // reuse Φ storage for w; Φ[r] is no longer needed
        quad += w * w;
      }

      const h = 5 * v;
      const b = Math.exp(params.hyp[h + 3]);
      const a = -params.hyp[h + 2] * b;
      const beta = Math.exp(x * params.hyp[h] + params.hyp[h + 1]) / (b * b);
      const s2 = 1 / beta + quad;

      const ys = (value - params.outscaler[4 * v]) / params.outscaler[4 * v + 1];
      const zv = (Math.sinh(b * Math.asinh(ys) - a) - mu) / Math.sqrt(s2);
      if (!Number.isFinite(zv)) continue;
      z[v] = zv;
      valid[v] = 1;
      if (age !== covariates.age) clamped++;
    }
  }

  return { score, result: (): ZmapResult => ({ z, valid, clamped }) };
}
