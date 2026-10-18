// Predictor: z-scores for one subject's map against one marker's voxel models.
// Two adapters (docs/adr/0001): "cpu" (float64, the reference) and "gpu"
// (WebGPU float32). createPredictor picks one and falls back to the CPU.
import type { Covariates, MarkerParams, ZmapResult } from "../model.ts";
import { zmapCpu } from "./kernel.ts";
import { createGpuZmap, supportsDesign, type GpuZmap } from "./webgpu.ts";

export type PredictorKind = "cpu" | "gpu";

export interface Predictor {
  kind: PredictorKind;
  zmap(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>): Promise<ZmapResult>;
  dispose(): void;
}

export function cpuPredictor(): Predictor {
  return { kind: "cpu", zmap: async (params, covariates, y) => zmapCpu(params, covariates, y), dispose() {} };
}

function gpuPredictor(gpu: GpuZmap): Predictor {
  return {
    kind: "gpu",
    // Stores whose design the shader does not cover are scored on the CPU.
    zmap: (params, covariates, y) => (supportsDesign(params) ? gpu.zmap(params, covariates, y) : Promise.resolve(zmapCpu(params, covariates, y))),
    dispose: () => gpu.dispose(),
  };
}

/** The preferred adapter, or the CPU one when WebGPU is unavailable; `reason` says why. */
export async function createPredictor(prefer: PredictorKind): Promise<{ predictor: Predictor; reason?: string }> {
  if (prefer === "cpu") return { predictor: cpuPredictor() };
  try {
    return { predictor: gpuPredictor(await createGpuZmap()) };
  } catch (error) {
    return { predictor: cpuPredictor(), reason: (error as Error).message };
  }
}
