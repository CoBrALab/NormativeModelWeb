// The whole app on a real subject: upload maps, enter covariates, compute on
// CPU and GPU, download the z-maps and compare them with
// packages/normative-dwi/validation/make_reference_zmap.py. Local data only;
// skipped unless NMZARR_STORE and NORMATIVE_E2E_DIR are set.
import { test, expect } from "@playwright/test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { gunzipSync } from "fflate";
import { serveStore } from "./serve-store.mjs";

const STORE = process.env.NMZARR_STORE;
const DIR = process.env.NORMATIVE_E2E_DIR;

function float32Volume(gz) {
  const bytes = gunzipSync(new Uint8Array(gz));
  const view = new DataView(bytes.buffer, bytes.byteOffset);
  const offset = view.getFloat32(108, true);
  return new Float32Array(bytes.slice(offset).buffer);
}

test.describe("whole subject", () => {
  test.skip(!STORE || !DIR, "set NMZARR_STORE and NORMATIVE_E2E_DIR");
  let server;
  test.beforeAll(async () => { server = await serveStore(STORE); });
  test.afterAll(() => server?.close());

  for (const compute of ["cpu", "gpu"]) {
    test(`${compute.toUpperCase()}: every z-map matches the reference`, async ({ page }, testInfo) => {
      test.setTimeout(600000);
      const { age, sex } = JSON.parse(readFileSync(join(DIR, "reference.json"), "utf8"));
      const maps = readdirSync(DIR).filter((f) => f.startsWith("input_")).map((f) => join(DIR, f));
      await page.goto("/");
      await page.locator("#imageInput").setInputFiles(maps);
      await expect(page.locator("#mapList select")).toHaveCount(maps.length);
      await page.locator("#age").fill(String(age));
      await page.locator("#sex").selectOption(sex);
      await page.locator("#advancedSettings > summary").click();
      await page.locator("#compute").selectOption(compute);
      await page.locator("#storeUrl").fill(`http://127.0.0.1:${server.address().port}/`);
      await page.locator("#storeUrl").dispatchEvent("change");
      await expect(page.locator("#runButton")).toBeEnabled();
      const started = Date.now();
      await page.locator("#runButton").click();
      await expect(page.locator("#statusText")).toContainText("z-map(s) ready", { timeout: 600000 });
      const seconds = (Date.now() - started) / 1000;
      await page.screenshot({ path: testInfo.outputPath(`app-${compute}.png`) });
      const status = await page.locator("#statusText").textContent();
      console.log(`${compute}: ${status} (wall ${seconds.toFixed(1)} s)`);
      expect(status).toContain(compute.toUpperCase());

      const rows = page.locator("#resultList").getByRole("button", { name: "Download", exact: true });
      await expect(rows).toHaveCount(maps.length);
      const tolerance = compute === "cpu" ? 1e-6 : 1e-3;
      for (let k = 0; k < maps.length; k++) {
        const download = page.waitForEvent("download");
        await rows.nth(k).click();
        const file = await download;
        const marker = file.suggestedFilename().replace("_zscore.nii.gz", "");
        const ours = float32Volume(readFileSync(await file.path()));
        const ref = float32Volume(readFileSync(join(DIR, `reference_${marker}.nii.gz`)));
        let worst = 0, zeroMismatch = 0;
        for (let i = 0; i < ref.length; i++) {
          if ((ref[i] === 0) !== (ours[i] === 0)) zeroMismatch++;
          worst = Math.max(worst, Math.abs(ours[i] - ref[i]) / Math.max(1, Math.abs(ref[i])));
        }
        console.log(`${compute} ${marker}: max rel |Δz| ${worst.toExponential(2)}, zero mismatches ${zeroMismatch}`);
        expect(zeroMismatch).toBe(0);
        expect(worst).toBeLessThan(tolerance);
      }
    });
  }
});
