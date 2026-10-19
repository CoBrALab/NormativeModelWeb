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
import { assertHardwareAdapter, hardwareGpu, readAdapterInfo } from "../../../test-utils/hardware-gpu.mjs";
import { storeVoxelAt, syntheticMap, syntheticModel, syntheticParams, writeSyntheticStore } from "../test/synthetic-store.js";
import { zmapCpu } from "../src/predictor/kernel.ts";
import { makeNifti } from "../test/nifti.js";

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
    if (compute === "gpu" && hardwareGpu) {
      // On the hardware-GPU runner the GPU must have scored, on a real (not software) adapter.
      expect(usedGpu).toBe(true);
      assertHardwareAdapter(await readAdapterInfo(page));
    }
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

test("a newer map of a marker replaces the older one", async ({ page }, testInfo) => {
  await prepare(page, "cpu");
  const again = testInfo.outputPath("sub-02_FA.nii");
  await writeFile(again, syntheticMap(model, "FA", 999).bytes);
  await page.locator("#imageInput").setInputFiles(again);
  await expect(page.locator("#mapList select")).toHaveCount(markers.length);
  await expect(page.locator("#mapList")).toContainText("sub-02_FA.nii");
  await expect(page.locator("#mapList")).not.toContainText("sub-01_FA.nii");
});

for (const layout of ["ras", "minc", "las"]) {
  test(`the readout gives the true z of the voxel under the crosshair (${layout} map)`, async ({ page }, testInfo) => {
    const map = syntheticMap(model, "FA", 31, layout);
    const path = testInfo.outputPath(`sub-03_${layout}_FA.nii`);
    await writeFile(path, map.bytes);
    await page.goto("./");
    await page.locator("#imageInput").setInputFiles(path);
    await page.locator("#age").fill(String(subject.age));
    await page.locator("#sex").selectOption(subject.sex);
    await page.locator("#advancedSettings > summary").click();
    await page.locator("#compute").selectOption("cpu");
    await page.locator("#storeUrl").fill(storeUrl);
    await page.locator("#storeUrl").dispatchEvent("change");
    await runToEnd(page);
    await expect(page.locator("#statusText")).not.toHaveClass(/error/);
    const { z, valid } = zmapCpu(syntheticParams(model, "FA"), subject, map.y);

    await page.locator("nd-viewer-toolbar").getByRole("button", { name: "Axial", exact: true }).click();
    const box = await page.locator("#gl1").boundingBox();
    let scored = 0;
    // NiiVue reports only moves of the crosshair, so each probe is a new position.
    for (const [dx, dy] of [[90, -70], [-110, 40], [30, 95], [-60, -100], [140, 10]]) {
      await page.mouse.click(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
      await expect(page.locator("#location")).toContainText("FA");
      const text = await page.locator("#location").textContent();
      const centre = text.match(/^(-?[\d.]+), (-?[\d.]+), (-?[\d.]+) mm/).slice(1).map(Number);
      const v = storeVoxelAt(model, centre);
      if (v < 0 || !valid[v]) {
        expect(text, `${layout} at ${centre}`).toContain("FA not scored");
      } else {
        expect(Number(text.match(/z = (-?[\d.]+)/)[1]), `${layout} at ${centre}`).toBeCloseTo(z[v], 3);
        scored++;
      }
    }
    expect(scored, "some probes land on scored voxels").toBeGreaterThan(0);
  });
}

/** A 1 mm structural over the model's box (its 2 mm grid spans the same voxel centres), shifted by `offset` mm. */
function syntheticStructural(offset = 0) {
  const [n0, n1, n2] = model.shape;
  const dims = [2 * n2 - 1, 2 * n1 - 1, 2 * n0 - 1];
  const data = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let z = 0; z < dims[2]; z++) for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
    data[x + dims[0] * (y + dims[1] * z)] = 100 + 50 * Math.sin(x / 3) * Math.cos(y / 4) + z;
  }
  return makeNifti(dims, [[1, 0, 0, -12 + offset], [0, 1, 0, -14], [0, 0, 1, -12], [0, 0, 0, 1]], data);
}

test("a structural background sits under the z-map, and the readout still gives the true z", async ({ page }, testInfo) => {
  const path = testInfo.outputPath("sub-01_T1w.nii");
  await writeFile(path, syntheticStructural());
  await prepare(page, "cpu");
  await runToEnd(page);
  await expect(page.locator("#statusText")).toContainText("z-map(s) ready");
  const canvas = page.locator("#gl1");
  const alone = await canvas.screenshot();

  await page.locator("#backgroundInput").setInputFiles(path);
  await expect(page.locator("#backgroundInfo")).toContainText("sub-01_T1w.nii · 23×27×23 at 1×1×1 mm");
  await expect(page.locator("#statusText")).toContainText("Showing FA z-map over sub-01_T1w.nii");
  await expect(page.locator("#statusText")).not.toHaveClass(/error/);
  expect(Buffer.compare(await canvas.screenshot(), alone), "the background changes the view").not.toBe(0);

  const { z, valid } = zmapCpu(syntheticParams(model, "FA"), subject, maps.FA.y);
  await page.locator("nd-viewer-toolbar").getByRole("button", { name: "Axial", exact: true }).click();
  const box = await canvas.boundingBox();
  let scored = 0;
  for (const [dx, dy] of [[90, -70], [-110, 40], [30, 95], [-60, -100], [140, 10]]) {
    await page.mouse.click(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
    await expect(page.locator("#location")).toContainText("FA");
    const text = await page.locator("#location").textContent();
    const v = storeVoxelAt(model, text.match(/^(-?[\d.]+), (-?[\d.]+), (-?[\d.]+) mm/).slice(1).map(Number));
    if (v < 0 || !valid[v]) continue;
    expect(Number(text.match(/z = (-?[\d.]+)/)[1])).toBeCloseTo(z[v], 3);
    scored++;
  }
  expect(scored, "some probes land on scored voxels").toBeGreaterThan(0);

  // Other markers keep the background.
  await page.locator("#resultList").getByRole("button", { name: "View", exact: true }).nth(1).click();
  await expect(page.locator("#statusText")).toContainText("z-map over sub-01_T1w.nii");

  await page.locator("#clearBackgroundButton").click();
  await expect(page.locator("#backgroundInfo")).toBeHidden();
  await expect(page.locator("#statusText")).toContainText("z-map · colour saturates");
});

test("a background outside the maps' field of view is refused", async ({ page }, testInfo) => {
  const path = testInfo.outputPath("elsewhere_T1w.nii");
  await writeFile(path, syntheticStructural(200));
  await page.goto("./");
  await page.locator("#imageInput").setInputFiles(maps.FA.path);
  await page.locator("#backgroundInput").setInputFiles(path);
  await expect(page.locator("#statusText")).toHaveClass(/error/);
  await expect(page.locator("#statusText")).toContainText("elsewhere_T1w.nii: its field of view covers only 0% of the z-map grid");
  await expect(page.locator("#backgroundInfo")).toBeHidden();
});

test("a background is shown on its own before any run", async ({ page }, testInfo) => {
  const path = testInfo.outputPath("sub-01_FLAIR.nii");
  await writeFile(path, syntheticStructural());
  await page.goto("./");
  await page.locator("#backgroundInput").setInputFiles(path);
  await expect(page.locator("#statusText")).toContainText("Background sub-01_FLAIR.nii loaded");
  await expect(page.locator("#emptyState")).toBeHidden();
  await page.locator("#clearBackgroundButton").click();
  await expect(page.locator("#emptyState")).toBeVisible();
});

test.describe("persistent cache", () => {
  // Playwright cannot route requests a service worker makes; the preview server
  // sends the isolation headers itself, so the isolation worker is not needed.
  test.use({ serviceWorkers: "block" });

  test("a store at a pinned Hugging Face URL is read from the browser cache on the next visit", async ({ page }) => {
    const base = `https://huggingface.co/datasets/test/webapps/resolve/${"c".repeat(40)}/store/`;
    let requests = 0;
    await page.route(`${base}**`, async (route) => {
      requests++;
      const request = route.request();
      const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "Content-Range, Content-Length", "Cross-Origin-Resource-Policy": "cross-origin" };
      let bytes;
      try {
        bytes = await readFile(join(dir, "store", decodeURIComponent(new URL(request.url()).pathname.slice(new URL(base).pathname.length))));
      } catch {
        return route.fulfill({ status: 404, headers: cors });
      }
      const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers().range ?? "");
      if (!range) return route.fulfill({ status: 200, headers: cors, body: bytes });
      const start = range[1] === "" ? Math.max(0, bytes.length - Number(range[2])) : Number(range[1]);
      const end = range[1] === "" || range[2] === "" ? bytes.length - 1 : Math.min(Number(range[2]), bytes.length - 1);
      return route.fulfill({ status: 206, headers: { ...cors, "Content-Range": `bytes ${start}-${end}/${bytes.length}` }, body: bytes.subarray(start, end + 1) });
    });
    const run = async () => {
      await page.goto("./");
      await page.locator("#imageInput").setInputFiles(maps.FA.path);
      await page.locator("#age").fill(String(subject.age));
      await page.locator("#sex").selectOption(subject.sex);
      await page.locator("#advancedSettings > summary").click();
      await page.locator("#compute").selectOption("cpu");
      await page.locator("#storeUrl").fill(base);
      await page.locator("#storeUrl").dispatchEvent("change");
      await runToEnd(page);
      await expect(page.locator("#statusText")).not.toHaveClass(/error/);
    };
    await run();
    const first = requests;
    expect(first).toBeGreaterThan(10);
    requests = 0;
    await run();
    expect(requests, "the second visit reads every chunk from Cache Storage").toBe(0);
    // Clearing the cache sends the next visit back to the network.
    await page.locator("#clearCacheButton").click();
    await expect(page.locator("#statusText")).toHaveText("Cached models cleared");
    await run();
    expect(requests).toBe(first);
  });
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
