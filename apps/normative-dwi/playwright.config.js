import { gpuBrowser } from "../../test-utils/hardware-gpu.mjs";
import { vitePreviewPlaywrightConfig } from "../../test-utils/playwright-vite-preview.mjs";

// Serve the BUILT output (skipped when CI provides it: NEURODESK_E2E_PREBUILT)
// so the shared preview headers and asset paths are exercised. gpuBrowser gives
// WebGPU on SwiftShader by default, and a real GPU with NEURODESK_HARDWARE_GPU=1
// (manual: the app declares no ci.hardware_gpu). e2e/gpu-parity.spec.js runs on
// the dev server: run it with playwright.gpu.config.js.
export default vitePreviewPlaywrightConfig({
  port: 4173,
  basePath: "/normative-dwi/",
  testIgnore: /gpu-parity\.spec\.js/,
  timeout: 600_000,
  use: { ...gpuBrowser },
});
