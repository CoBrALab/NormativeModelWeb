// WebGPU float32 adapter against PCNtoolkit and against the float64 CPU kernel
// (docs/adr/0001). Run with: pnpm exec playwright test -c playwright.gpu.config.js
//
// Always runs on the committed reference (PCNtoolkit's fits to simulated data,
// tools/make_synthetic_reference.py); the per-marker cases need the local
// UK Biobank-derived fixtures from tools/make_fixtures.py and skip without them.
import { readFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { FIXTURE_MARKERS, fixturesAvailable, loadFixture } from "../test/fixtures.js";

// Float32 tolerance, relative to max(1, |z|). Measured values are printed and
// recorded in docs/adr/0001.
const GPU_TOLERANCE = 1e-3;

const loadReference = () => JSON.parse(readFileSync(new URL("../test/reference/synthetic-pcn.json", import.meta.url), "utf8"));
const cases = [
  { name: "synthetic (committed reference)", load: loadReference, skip: false },
  ...FIXTURE_MARKERS.map((marker) => ({
    name: marker,
    load: () => loadFixture(marker),
    skip: fixturesAvailable() ? false : "no local fixtures (run tools/make_fixtures.py)",
  })),
];

for (const { name, load, skip } of cases) {
  test(`${name}: WebGPU z agrees with PCNtoolkit and the CPU kernel`, async ({ page }) => {
    test.skip(Boolean(skip), String(skip));
    await page.goto("./");
    const fixture = load();
    const result = await page.evaluate(async (fixture) => {
      const base = new URL("./", location.href).pathname;
      const { createGpuZmap } = await import(`${base}src/predictor/webgpu.ts`);
      const { zmapCpu } = await import(`${base}src/predictor/kernel.ts`);
      const { fixtureParams, sexOf } = await import(`${base}test/fixture-params.js`);
      const gpu = await createGpuZmap();
      let vsOracle = 0, vsCpu = 0, cases = 0, clampMismatch = 0;
      const started = performance.now();
      for (const record of fixture.voxels) {
        const params = fixtureParams(fixture, [record]);
        for (const c of record.cases) {
          const cov = { age: c.age, sex: sexOf(c.sex) };
          const g = await gpu.zmap(params, cov, [c.y]);
          const cpu = zmapCpu(params, cov, [c.y]);
          const scale = Math.max(1, Math.abs(c.z));
          vsOracle = Math.max(vsOracle, Math.abs(g.z[0] - c.z) / scale);
          vsCpu = Math.max(vsCpu, Math.abs(g.z[0] - cpu.z[0]) / scale);
          if (g.clamped !== cpu.clamped || g.valid[0] !== cpu.valid[0]) clampMismatch++;
          cases++;
        }
      }
      gpu.dispose();
      return { vsOracle, vsCpu, cases, clampMismatch, ms: performance.now() - started };
    }, fixture);
    console.log(`${name}: ${result.cases} cases, max rel |Δz| GPU–PCN ${result.vsOracle.toExponential(2)}, GPU–CPU ${result.vsCpu.toExponential(2)}`);
    expect(result.cases).toBeGreaterThan(0);
    expect(result.clampMismatch).toBe(0);
    expect(result.vsCpu).toBeLessThan(GPU_TOLERANCE);
    expect(result.vsOracle).toBeLessThan(GPU_TOLERANCE);
  });
}
