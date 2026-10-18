import { defineConfig } from "@playwright/test";

// Predictor parity in a real browser: imports src/ modules from the Vite dev
// server and runs the WebGPU adapter (SwiftShader when no GPU is present).
export default defineConfig({
  testDir: "./e2e",
  testMatch: /gpu-parity\.spec\.js/,
  timeout: 600000,
  webServer: {
    command: "pnpm exec vite --port 4174 --strictPort",
    url: "http://localhost:4174/normative-dwi/",
    reuseExistingServer: !process.env.CI,
  },
  use: {
    baseURL: "http://localhost:4174/normative-dwi/",
    launchOptions: { args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] },
  },
});
