import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { gunzipSync } from "fflate";
import { gatherToStore, gridMapping, GridMismatchError, loadSubjectMap, scatterToNifti } from "../src/subject-image/index.ts";
import { code, makeNifti, smallGrid, worldCodedNifti, worldOf } from "./nifti.js";

const grid = smallGrid();
// The same template space written x-fastest (the usual NIfTI layout).
const xFirst = { dims: [6, 5, 4], affine: [[2, 0, 0, -10], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]] };

async function mapFor(dims, affine) {
  return loadSubjectMap("map.nii", worldCodedNifti(dims, affine));
}

function assertGathersWorld(map) {
  const y = gatherToStore(map, gridMapping(map, grid));
  for (let v = 0; v < grid.nVox; v++) assert.equal(y[v], Math.fround(code(worldOf(grid, v))), `voxel ${v}`);
}

test("a map written x-fastest is matched voxel by voxel through the affines", async () => {
  assertGathersWorld(await mapFor(xFirst.dims, xFirst.affine));
});

test("a map in store (MINC) axis order is matched too", async () => {
  assertGathersWorld(await mapFor([6, 5, 4].reverse(), [[0, 0, 2, -10], [0, 2, 0, -20], [2, 0, 0, -30], [0, 0, 0, 1]]));
});

test("a map with a flipped axis is matched too", async () => {
  // x runs right-to-left: voxel 0 sits at the far end.
  assertGathersWorld(await mapFor(xFirst.dims, [[-2, 0, 0, 0], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]]));
});

for (const [label, dims, affine] of [
  ["the wrong size", [6, 5, 3], xFirst.affine],
  ["1 mm voxels", [12, 10, 8], [[1, 0, 0, -10], [0, 1, 0, -20], [0, 0, 1, -30], [0, 0, 0, 1]]],
  ["a half-voxel shift", xFirst.dims, [[2, 0, 0, -9], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]]],
  ["a rotation", xFirst.dims, [[1.9, 0.6, 0, -10], [-0.6, 1.9, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]]],
]) {
  test(`a map with ${label} is rejected with the expected grid`, async () => {
    const map = await mapFor(dims, affine);
    assert.throws(() => gridMapping(map, grid), (error) => error instanceof GridMismatchError && /expected a 4×5×6 map at 2 mm in template space; found/.test(error.message));
  });
}

test("a 4-D image is rejected", async () => {
  const bytes = makeNifti(xFirst.dims, xFirst.affine, new Float32Array(6 * 5 * 4 * 2));
  new DataView(bytes.buffer).setInt16(40, 4, true);
  new DataView(bytes.buffer).setInt16(48, 2, true);
  await assert.rejects(loadSubjectMap("dwi.nii", bytes), /expected one 3-D map, found 2 volumes/);
});

test("the z-map is written on the map's own grid, 0 where not scored", async () => {
  const map = await mapFor(xFirst.dims, xFirst.affine);
  const mapping = gridMapping(map, grid);
  const z = Float64Array.from({ length: grid.nVox }, (_, v) => v / 7 - 2);
  const valid = Uint8Array.from({ length: grid.nVox }, (_, v) => (v % 4 ? 1 : 0));
  const bytes = scatterToNifti(z, valid, mapping, map);
  const back = await loadSubjectMap("z.nii.gz", gunzipSync(bytes));
  assert.deepEqual(back.header.affine.map((r) => [...r]), map.header.affine.map((r) => [...r]));
  assert.equal(back.header.datatype, 16);
  const expected = new Float32Array(6 * 5 * 4);
  for (let v = 0; v < grid.nVox; v++) if (valid[v]) expected[mapping[v]] = z[v];
  assert.deepEqual(Float32Array.from(back.data), expected);
});

const TEMPLATE = "/scratch/parent41/UKB/temporary_template/avg.020_2mm.nii.gz";
const STORE = process.env.NMZARR_STORE;
test("the UK Biobank 2 mm template NIfTI sits on the store's grid", { skip: !(STORE && existsSync(TEMPLATE)) && "needs NMZARR_STORE and the template" }, async () => {
  const { openModelStore } = await import("../src/model-store/index.ts");
  const { fsDirectory } = await import("./fs-directory.js");
  const store = await openModelStore({ kind: "directory", handle: fsDirectory(STORE) });
  const map = await loadSubjectMap("template", readFileSync(TEMPLATE));
  const y = gatherToStore(map, gridMapping(map, store.grid));
  const inBrain = y.filter((value) => value > 20).length;
  assert.ok(inBrain / y.length > 0.6, `only ${inBrain} of ${y.length} mask voxels look like brain`);
});

test("the display copy saturates z and puts unscored voxels below the display range", async () => {
  const { displayNifti } = await import("../src/subject-image/index.ts");
  const map = await mapFor(xFirst.dims, xFirst.affine);
  const mapping = gridMapping(map, grid);
  const z = Float64Array.from({ length: grid.nVox }, (_, v) => [-7, -3, 0.5, 3, 9][v % 5]);
  const valid = Uint8Array.from({ length: grid.nVox }, (_, v) => (v % 6 ? 1 : 0));
  const back = await loadSubjectMap("display.nii", displayNifti(z, valid, mapping, map, 3));
  for (let v = 0; v < grid.nVox; v++) {
    const shown = back.data[mapping[v]];
    if (valid[v]) assert.equal(shown, Math.min(3, Math.max(-3, z[v])));
    else assert.ok(shown < -3);
  }
});
