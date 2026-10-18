import { defineConfig } from "@playwright/test";

// Serve the BUILT output so the shared preview headers and worker/wasm asset
// paths are exercised — not just the dev server.
export default defineConfig({
  testDir: "./e2e",
  webServer: {
    command: "pnpm build && pnpm preview --port 4173 --strictPort",
    url: "http://localhost:4173/normative-dwi/",
    reuseExistingServer: !process.env.CI,
  },
  testIgnore: /gpu-parity\.spec\.js/,
  use: {
    baseURL: "http://localhost:4173/normative-dwi/",
    launchOptions: { args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] },
  },
});
