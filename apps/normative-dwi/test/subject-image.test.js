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

const c30 = 2 * Math.cos(Math.PI / 6), s30 = 2 * Math.sin(Math.PI / 6);
for (const [label, dims, affine, reason] of [
  ["the wrong size", [6, 5, 3], xFirst.affine, /its size is 6×5×3/],
  ["1 mm voxels", xFirst.dims, [[1, 0, 0, -10], [0, 1, 0, -20], [0, 0, 1, -30], [0, 0, 0, 1]], /its voxels are 1×1×1 mm/],
  ["a half-voxel shift", xFirst.dims, [[2, 0, 0, -9], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]], /its origin is off the template's voxel centres by -1×0×0 mm/],
  ["a whole-voxel shift", xFirst.dims, [[2, 0, 0, -6], [0, 2, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]], /its origin is shifted by whole voxels/],
  ["a rotation", xFirst.dims, [[c30, -s30, 0, -10], [s30, c30, 0, -20], [0, 0, 2, -30], [0, 0, 0, 1]], /rotated or oblique/],
]) {
  test(`a map with ${label} is rejected and says what differs`, async () => {
    const map = await mapFor(dims, affine);
    assert.throws(() => gridMapping(map, grid), (error) => error instanceof GridMismatchError
      && /map\.nii: not on the 4×5×6 template grid at 2×2×2 mm; /.test(error.message) && reason.test(error.message));
  });
}

test("an affine in metres (xyzt_units) is read in millimetres", async () => {
  const metres = xFirst.affine.map((row, r) => (r < 3 ? row.map((v) => v / 1000) : row));
  const bytes = worldCodedNifti(xFirst.dims, xFirst.affine);
  const view = new DataView(bytes.buffer);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) view.setFloat32(280 + 16 * r + 4 * c, metres[r][c], true);
  view.setUint8(123, 1); // NIFTI_UNITS_METER
  assertGathersWorld(await loadSubjectMap("metres.nii", bytes));
});

test("the readout locates voxels in the raw affine's units, as NiiVue reports them", async () => {
  const { voxelAtMm } = await import("../src/subject-image/index.ts");
  const metres = xFirst.affine.map((row, r) => (r < 3 ? row.map((v) => v / 1000) : row));
  const bytes = worldCodedNifti(xFirst.dims, xFirst.affine);
  const view = new DataView(bytes.buffer);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) view.setFloat32(280 + 16 * r + 4 * c, metres[r][c], true);
  view.setUint8(123, 1); // NIFTI_UNITS_METER
  const map = await loadSubjectMap("metres.nii", bytes);
  const locate = voxelAtMm(map.header);
  const [i, j, k] = [3, 1, 2];
  const world = [0, 1, 2].map((r) => metres[r][0] * i + metres[r][1] * j + metres[r][2] * k + metres[r][3]);
  assert.equal(locate(world)?.index, i + xFirst.dims[0] * (j + xFirst.dims[1] * k));
});

test("a file that is not NIfTI-1 is refused", async () => {
  await assert.rejects(loadSubjectMap("notes.txt", new TextEncoder().encode("x".repeat(400))), /notes\.txt: not a NIfTI-1 image/);
});

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
  const bytes = scatterToNifti(z, valid, mapping, map, "normative z (test FA)");
  const raw = gunzipSync(bytes);
  const header = new DataView(raw.buffer, raw.byteOffset);
  assert.equal(header.getInt16(68, true), 5, "intent NIFTI_INTENT_ZSCORE");
  assert.equal(header.getFloat32(108, true), 352, "no extensions");
  assert.equal(new TextDecoder().decode(raw.subarray(148, 148 + 21)), "normative z (test FA)");
  const back = await loadSubjectMap("z.nii.gz", raw);
  assert.deepEqual(back.header.affine.map((r) => [...r]), map.header.affine.map((r) => [...r]));
  assert.equal(back.header.datatype, 16);
  const expected = new Float32Array(6 * 5 * 4);
  for (let v = 0; v < grid.nVox; v++) if (valid[v]) expected[mapping[v]] = z[v];
  assert.deepEqual(Float32Array.from(back.data), expected);
});

// A real template-space image, e.g. proc_nm's templates/tissue/avg.020_2mm (as NIfTI).
const TEMPLATE = process.env.NORMATIVE_TEMPLATE_NIFTI;
const STORE = process.env.NMZARR_STORE;
test("a UK Biobank 2 mm template NIfTI sits on the store's grid", { skip: !(STORE && TEMPLATE && existsSync(TEMPLATE)) && "needs NMZARR_STORE and NORMATIVE_TEMPLATE_NIFTI" }, async () => {
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

test("the map voxel at a world position is found in any axis order or direction", async () => {
  const { voxelAtMm } = await import("../src/subject-image/index.ts");
  const { syntheticMap, syntheticModel, storeVoxelAt } = await import("./synthetic-store.js");
  const model = syntheticModel({ markers: ["FA"] });
  for (const layout of ["ras", "minc", "las"]) {
    const { bytes, y } = syntheticMap(model, "FA", 3, layout);
    const map = await loadSubjectMap(`${layout}.nii`, bytes);
    const locate = voxelAtMm(map.header);
    for (const v of [0, 17, 400, model.nVox - 1]) {
      const [i, j, k] = model.ijk.subarray(3 * v, 3 * v + 3);
      const world = [2 * k - 12, 2 * j - 14, 2 * i - 12];
      const found = locate(world.map((c) => c + 0.6)); // inside the voxel, off its centre
      assert.deepEqual(found.centre, world, `${layout} voxel ${v}`);
      assert.equal(map.data[found.index], Math.fround(y[v]), `${layout} voxel ${v}`);
      assert.equal(storeVoxelAt(model, found.centre), v);
    }
    assert.equal(locate([500, 0, 0]), null);
  }
});
