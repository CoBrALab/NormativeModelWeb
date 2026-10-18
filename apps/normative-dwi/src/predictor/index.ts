// Predictor: z-scores for one subject's map against one marker's voxel models.
// Two adapters (docs/adr/0001): "cpu" (float64, the reference) and "gpu"
// (WebGPU float32). createPredictor picks one; a GPU that fails or is lost
// hands the marker, and every later one, to the CPU.
import type { Covariates, MarkerParams, ZmapResult } from "../model.ts";
import { sexColumn, zmapCpuAsync } from "./kernel.ts";
import { createGpuZmap, supportsDesign, type GpuZmap } from "./webgpu.ts";

export type PredictorKind = "cpu" | "gpu";

export interface Predictor {
  /** The adapter that will score the next call (a failed GPU becomes "cpu"). */
  readonly kind: PredictorKind;
  zmap(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal): Promise<ZmapResult & { kind: PredictorKind }>;
  dispose(): void;
}

export function cpuPredictor(): Predictor {
  return {
    kind: "cpu",
    zmap: async (params, covariates, y, signal) => ({ ...(await zmapCpuAsync(params, covariates, y, signal)), kind: "cpu" }),
    dispose() {},
  };
}

/** GPU first; on a GPU error or a lost device, warn once and continue on the CPU. */
export function gpuPredictor(gpu: GpuZmap, onFallback: (reason: string) => void = () => {}): Predictor {
  let failed: string | null = null;
  const fallBack = (reason: string) => {
    if (failed) return;
    failed = reason;
    gpu.dispose();
    onFallback(reason);
  };
  return {
    get kind() {
      return failed || gpu.lostReason() ? "cpu" : "gpu";
    },
    async zmap(params, covariates, y, signal) {
      // Input errors are not GPU failures: raise them before the GPU is tried,
      // so they cannot switch the session to the CPU.
      if (y.length !== params.nVox) throw new Error(`Expected ${params.nVox} values, got ${y.length}`);
      sexColumn(params, covariates);
      const lost = gpu.lostReason();
      if (lost) fallBack(lost);
      if (!failed && supportsDesign(params)) {
        try {
          return { ...(await gpu.zmap(params, covariates, y, signal)), kind: "gpu" };
        } catch (error) {
          if ((error as Error).name === "AbortError") throw error;
          fallBack((error as Error).message);
        }
      }
      return { ...(await zmapCpuAsync(params, covariates, y, signal)), kind: "cpu" };
    },
    dispose: () => {
      if (!failed) gpu.dispose();
    },
  };
}

/** The preferred adapter, or the CPU one when WebGPU is unavailable; `reason` says why. */
export async function createPredictor(prefer: PredictorKind, onFallback?: (reason: string) => void): Promise<{ predictor: Predictor; reason?: string }> {
  if (prefer === "cpu") return { predictor: cpuPredictor() };
  try {
    return { predictor: gpuPredictor(await createGpuZmap(), onFallback) };
  } catch (error) {
    return { predictor: cpuPredictor(), reason: (error as Error).message };
  }
}
