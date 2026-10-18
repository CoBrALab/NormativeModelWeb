import { vitePreviewPlaywrightConfig } from "../../test-utils/playwright-vite-preview.mjs";

// Serve the BUILT output (skipped when CI provides it: NEURODESK_E2E_PREBUILT)
// so the shared preview headers and asset paths are exercised. WebGPU is
// enabled so the GPU predictor runs where the browser has an adapter.
// e2e/gpu-parity.spec.js needs local UK Biobank-derived fixtures and the dev
// server: run it with playwright.gpu.config.js.
export default vitePreviewPlaywrightConfig({
  port: 4173,
  basePath: "/normative-dwi/",
  testIgnore: /gpu-parity\.spec\.js/,
  use: { launchOptions: { args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] } },
});
