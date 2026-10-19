import type { Covariates, MarkerParams, ZmapResult } from "../model.js";

export type PredictorKind = "cpu" | "gpu";

export interface Predictor {
  /** The adapter that will score the next call (a failed GPU becomes "cpu"). */
  readonly kind: PredictorKind;
  zmap(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal): Promise<ZmapResult & { kind: PredictorKind }>;
  dispose(): void;
}

/** The float64 reference adapter. */
export function cpuPredictor(): Predictor;
