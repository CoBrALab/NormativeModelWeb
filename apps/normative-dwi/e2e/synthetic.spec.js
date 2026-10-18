// The whole app against a generated store with random parameters (no data
// needed, so it runs on every CI job): upload, score on CPU and GPU, download,
// and compare with the float64 kernel run in Node on the same parameters.
// With NEURODESK_HARDWARE_GPU=1 (the macOS GPU runner) the GPU run must really
// use the GPU; elsewhere the app may fall back to the CPU.
import { test, expect } from "@playwright/test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "fflate";
import { serveStore } from "./serve-store.mjs";
import { syntheticMap, syntheticModel, syntheticParams, writeSyntheticStore } from "../test/synthetic-store.js";
import { zmapCpu } from "../src/predictor/kernel.ts";

const model = syntheticModel();
const subject = { age: 66.5, sex: "female" };
const markers = ["FA", "MD", "OD"];
let dir;
let server;
let storeUrl;
const maps = {};

test.beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "normative-dwi-e2e-"));
  await writeSyntheticStore(join(dir, "store"), model);
  for (const [k, marker] of markers.entries()) {
    const { bytes, y } = syntheticMap(model, marker, 100 + k);
    const path = join(dir, `sub-01_${marker}.nii`);
    await writeFile(path, bytes);
    maps[marker] = { path, y };
  }
  server = await serveStore(join(dir, "store"));
  storeUrl = `http://127.0.0.1:${server.address().port}/`;
});
test.afterAll(async () => {
  server?.close();
  if (dir) await rm(dir, { recursive: true, force: true });
});

/** Float32 voxels of a downloaded .nii.gz, and its header view. */
async function readZmap(download) {
  const bytes = gunzipSync(new Uint8Array(await readFile(await download.path())));
  const header = new DataView(bytes.buffer, bytes.byteOffset);
  return { header, data: new Float32Array(bytes.slice(header.getFloat32(108, true)).buffer) };
}

/** Expected z on the map's own (x-fastest) grid, 0 where not scored. */
function expectedVolume(marker) {
  const { z, valid } = zmapCpu(syntheticParams(model, marker), subject, maps[marker].y);
  const [n0, n1, n2] = model.shape;
  const volume = new Float32Array(n0 * n1 * n2);
  for (let v = 0; v < model.nVox; v++) {
    const [i, j, k] = model.ijk.subarray(3 * v, 3 * v + 3);
    if (valid[v]) volume[k + n2 * (j + n1 * i)] = z[v];
  }
  return volume;
}

async function prepare(page, compute) {
  await page.goto("./");
  await page.locator("#imageInput").setInputFiles(markers.map((m) => maps[m].path));
  await expect(page.locator("#mapList select")).toHaveCount(markers.length);
  await page.locator("#age").fill(String(subject.age));
  await page.locator("#sex").selectOption(subject.sex);
  await page.locator("#advancedSettings > summary").click();
  await page.locator("#compute").selectOption(compute);
  await page.locator("#storeUrl").fill(storeUrl);
  await page.locator("#storeUrl").dispatchEvent("change");
  await expect(page.locator("#runButton")).toBeEnabled();
}

async function runToEnd(page) {
  await page.locator("#runButton").click();
  await page.waitForFunction(() => {
    const status = document.getElementById("statusText");
    return status.classList.contains("error") || /ready|Showing/.test(status.textContent);
  }, null, { timeout: 120000 });
}

for (const compute of ["cpu", "gpu"]) {
  test(`${compute.toUpperCase()}: z-maps match the float64 kernel on a generated store`, async ({ page }) => {
    await prepare(page, compute);
    await runToEnd(page);
    await expect(page.locator("#statusText")).not.toHaveClass(/error/);
    const log = await page.locator("#technicalLog").textContent();
    const usedGpu = /scored on GPU/.test(log);
    if (compute === "gpu" && process.env.NEURODESK_HARDWARE_GPU === "1") expect(usedGpu).toBe(true);
    if (compute === "cpu") expect(usedGpu).toBe(false);

    const rows = page.locator("#resultList").getByRole("button", { name: "Download", exact: true });
    await expect(rows).toHaveCount(markers.length);
    for (let k = 0; k < markers.length; k++) {
      const pending = page.waitForEvent("download");
      await rows.nth(k).click();
      const download = await pending;
      const marker = download.suggestedFilename().replace("_zscore.nii.gz", "");
      const { header, data } = await readZmap(download);
      expect(header.getInt16(68, true)).toBe(5);
      const expected = expectedVolume(marker);
      let worst = 0, zeroMismatch = 0;
      for (let i = 0; i < expected.length; i++) {
        if ((expected[i] === 0) !== (data[i] === 0)) zeroMismatch++;
        worst = Math.max(worst, Math.abs(data[i] - expected[i]) / Math.max(1, Math.abs(expected[i])));
      }
      console.log(`${compute}${usedGpu ? "" : " (cpu)"} ${marker}: max rel |Δz| ${worst.toExponential(2)}`);
      expect(zeroMismatch).toBe(0);
      expect(worst).toBeLessThan(usedGpu ? 1e-3 : 1e-6);
    }
  });
}

test("a local store folder works like a store URL", async ({ page }) => {
  await page.goto("./");
  await page.locator("#imageInput").setInputFiles(maps.FA.path);
  await page.locator("#age").fill(String(subject.age));
  await page.locator("#sex").selectOption(subject.sex);
  await page.locator("#advancedSettings > summary").click();
  await page.locator("#compute").selectOption("cpu");
  await page.locator("#storeFolder").setInputFiles(join(dir, "store"));
  await expect(page.locator("#storeInfo")).toContainText("store ·");
  await runToEnd(page);
  await expect(page.locator("#statusText")).not.toHaveClass(/error/);
  const pending = page.waitForEvent("download");
  await page.locator("#resultList").getByRole("button", { name: "Download", exact: true }).click();
  const { data } = await readZmap(await pending);
  const expected = expectedVolume("FA");
  expect(data.every((value, i) => value === expected[i])).toBe(true);
});

test("a map on the wrong grid is skipped and named; the others are still scored", async ({ page }, testInfo) => {
  const shifted = syntheticMap(model, "ICVF", 7).bytes;
  new DataView(shifted.buffer).setFloat32(280 + 12, -11, true); // half a voxel along x
  const path = testInfo.outputPath("sub-01_ICVF.nii");
  await writeFile(path, shifted);
  await prepare(page, "cpu");
  await page.locator("#imageInput").setInputFiles(path);
  await expect(page.locator("#mapList select")).toHaveCount(markers.length + 1);
  await runToEnd(page);
  await expect(page.locator("#statusText")).toContainText("skipped ICVF");
  await expect(page.locator("#mapList")).toContainText("its origin is off the template's voxel centres");
  await expect(page.locator("#resultList").getByRole("button", { name: "Download", exact: true })).toHaveCount(markers.length);
});

test("a newer map of a marker replaces the older one, and the readout reports its true z", async ({ page }, testInfo) => {
  await prepare(page, "cpu");
  const again = testInfo.outputPath("sub-02_FA.nii");
  const second = syntheticMap(model, "FA", 999);
  await writeFile(again, second.bytes);
  await page.locator("#imageInput").setInputFiles(again);
  await expect(page.locator("#mapList select")).toHaveCount(markers.length);
  await expect(page.locator("#mapList")).toContainText("sub-02_FA.nii");
  await expect(page.locator("#mapList")).not.toContainText("sub-01_FA.nii");
  await runToEnd(page);

  await page.locator("#resultList").getByText("FA z-map").locator("xpath=..").getByRole("button", { name: "View", exact: true }).click();
  await expect(page.locator("#statusText")).toContainText("Showing FA");
  // Click off the starting crosshair (NiiVue reports only moves); the readout
  // must give the unsaturated z of the voxel it names.
  await page.locator("nd-viewer-toolbar").getByRole("button", { name: "Axial", exact: true }).click();
  const box = await page.locator("#gl1").boundingBox();
  await page.mouse.click(box.x + box.width / 2 + 90, box.y + box.height / 2 - 70);
  await expect(page.locator("#location")).toContainText(/FA (z = -?\d|not scored)/);
  const text = await page.locator("#location").textContent();
  // The map is x-fastest: file voxel (a, b, c) is store voxel (i, j, k) = (c, b, a).
  const [a, b, c] = text.match(/voxel (\d+),(\d+),(\d+)/).slice(1).map(Number);
  const v = [...Array(model.nVox).keys()].find((n) => model.ijk[3 * n] === c && model.ijk[3 * n + 1] === b && model.ijk[3 * n + 2] === a);
  const { z: zs, valid } = zmapCpu(syntheticParams(model, "FA"), subject, second.y);
  if (v === undefined || !valid[v]) expect(text).toContain("FA not scored");
  else expect(Number(text.match(/z = (-?[\d.]+)/)[1])).toBeCloseTo(zs[v], 2);
});

test("the automation operation scores maps from a given store", async ({ page }) => {
  await page.goto("./");
  await page.locator("#neurodesk-input-transfer").setInputFiles(markers.map((m) => maps[m].path));
  const report = await page.evaluate(async (storeUrl) => {
    const automation = globalThis.neurodeskAutomation;
    await automation.dispatch("adopt", { role: "maps" });
    await automation.dispatch("start", { parameters: { age: 66.5, sex: "female", compute: "cpu", storeUrl } });
    for (let i = 0; i < 600; i++) {
      const snapshot = await automation.dispatch("snapshot");
      if (snapshot.state !== "running") return snapshot;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    return automation.dispatch("snapshot");
  }, storeUrl);
  expect(report.state).toBe("succeeded");
  expect(JSON.stringify(report.report.artifacts)).toContain("zmap");
});
