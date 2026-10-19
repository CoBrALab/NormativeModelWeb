// Predictor: z-scores for one subject's map against one marker's voxel models.
// This is the float64 CPU adapter, the reference (apps/normative-dwi/docs/adr/0001);
// the web app adds a WebGPU adapter with the same interface.
import { zmapCpuAsync } from "./kernel.js";

export function cpuPredictor() {
  return {
    kind: "cpu",
    zmap: async (params, covariates, y, signal) => ({ ...(await zmapCpuAsync(params, covariates, y, signal)), kind: "cpu" }),
    dispose() {},
  };
}
