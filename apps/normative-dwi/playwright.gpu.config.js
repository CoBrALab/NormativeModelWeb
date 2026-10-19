import { defineConfig } from "@playwright/test";
import { gpuBrowser } from "../../test-utils/hardware-gpu.mjs";

// Predictor parity in a real browser: imports src/ modules from the Vite dev
// server and runs the WebGPU adapter against the committed PCNtoolkit reference
// and, when present, the local fixtures (SwiftShader by default; a real GPU
// with NEURODESK_HARDWARE_GPU=1).
export default defineConfig({
  testDir: "./e2e",
  testMatch: /gpu-parity\.spec\.js/,
  timeout: 600_000,
  webServer: {
    command: "pnpm exec vite --port 4174 --strictPort",
    url: "http://localhost:4174/normative-dwi/",
    reuseExistingServer: !process.env.CI,
  },
  use: { baseURL: "http://localhost:4174/normative-dwi/", ...gpuBrowser },
});
