// ModelStore against the real nmzarr store on disk, through the local-folder
// adapter (a fake directory handle backed by node:fs). Skipped unless
// NMZARR_STORE points at a store, e.g.
//   NMZARR_STORE=/scratch/parent41/proc_nm/opt_inference/stores/ukb_blr_agesex.zarr
import assert from "node:assert/strict";
import test from "node:test";
import { openModelStore } from "../src/model-store/index.ts";
import { directoryStore } from "../src/model-store/directory-store.ts";
import { zmapCpu } from "../src/predictor/kernel.ts";
import { FIXTURE_MARKERS, fixturesAvailable, loadFixture, sexOf } from "./fixtures.js";
import { fsDirectory, notFound } from "./fs-directory.js";

const STORE = process.env.NMZARR_STORE;
const skip = STORE ? false : "set NMZARR_STORE to run against a real store";
const skipParity = skip || (fixturesAvailable() ? false : "no local fixtures (run tools/make_fixtures.py)");

test("opens the store and exposes template space", { skip }, async () => {
  const store = await openModelStore({ kind: "directory", handle: fsDirectory(STORE) });
  assert.equal(store.family, "ukb_blr_agesex");
  assert.deepEqual([...store.markers].sort(), [...FIXTURE_MARKERS].sort());
  assert.deepEqual(store.grid.shape, [97, 115, 97]);
  assert.equal(store.grid.nVox, 304327);
  assert.deepEqual([...store.grid.affine], [0, 0, 2, -96, 0, 2, 0, -132, 2, 0, 0, -78, 0, 0, 0, 1]);
});

for (const marker of FIXTURE_MARKERS) {
  test(`${marker}: parameters read from the store reproduce the PCNtoolkit fixtures`, { skip: skipParity, timeout: 300000 }, async () => {
    const store = await openModelStore({ kind: "directory", handle: fsDirectory(STORE) });
    const fractions = [];
    const params = await store.loadMarker(marker, { onProgress: (f) => fractions.push(f) });
    assert.equal(params.nVox, store.grid.nVox);
    assert.ok(fractions.length > 9 && Math.abs(fractions.at(-1) - 1) < 1e-12);

    const fixture = loadFixture(marker);
    let worst = 0;
    for (const record of fixture.voxels) {
      const v = record.voxel;
      assert.equal(params.hasModel[v], 1);
      assert.deepEqual([...params.m.subarray(v * 9, v * 9 + 9)], record.params.m);
      for (const c of record.cases.filter((_, k) => k % 8 === 0)) {
        const y = new Float64Array(params.nVox);
        y[v] = c.y;
        const { z, valid } = zmapCpu(params, { age: c.age, sex: sexOf(c.sex) }, y);
        assert.equal(valid[v], 1);
        worst = Math.max(worst, Math.abs(z[v] - c.z));
      }
    }
    for (const v of fixture.unmodelled_voxels) assert.equal(params.hasModel[v], 0);
    assert.ok(worst < 1e-6, `max |Δz| ${worst}`);
  });
}

test("a folder that is not a model store is rejected clearly", async () => {
  const empty = { name: "empty", getDirectoryHandle: async () => { throw notFound(); }, getFileHandle: async () => { throw notFound(); } };
  await assert.rejects(openModelStore({ kind: "directory", handle: empty }), /empty is not a readable model store/);
});

test("directory store reads suffix and offset ranges", { skip }, async () => {
  const store = directoryStore(fsDirectory(STORE));
  const whole = await store.get("/zarr.json");
  const tail = await store.getRange("/zarr.json", { suffixLength: 10 });
  const middle = await store.getRange("/zarr.json", { offset: 5, length: 7 });
  assert.deepEqual(tail, whole.subarray(whole.length - 10));
  assert.deepEqual(middle, whole.subarray(5, 12));
  assert.equal(await store.get("/no/such/key"), undefined);
});
