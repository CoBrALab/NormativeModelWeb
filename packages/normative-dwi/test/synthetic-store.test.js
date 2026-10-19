// ModelStore on a generated store: reading, and the design and layout checks.
// Needs no data.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { openModelStore } from "../src/model-store/index.js";
import { folderDirectory } from "../src/node.js";
import { readdir } from "node:fs/promises";
import { SHARD_VOXELS, SYNTHETIC_FAMILY, syntheticModel, writeSyntheticStore } from "../src/synthetic-store.js";

const model = syntheticModel();

async function withStore(edit, run) {
  const dir = await mkdtemp(join(tmpdir(), "nmzarr-"));
  try {
    await writeSyntheticStore(dir, model);
    if (edit) await edit(dir);
    return await run(dir);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

async function editJson(path, change) {
  const doc = JSON.parse(await readFile(path, "utf8"));
  change(doc);
  await writeFile(path, JSON.stringify(doc));
}

test("a generated store opens and returns its parameters exactly", () =>
  withStore(null, async (dir) => {
    const store = await openModelStore({ kind: "directory", handle: folderDirectory(dir) });
    assert.equal(store.family, SYNTHETIC_FAMILY);
    assert.deepEqual(store.grid.shape, model.shape);
    assert.equal(store.grid.nVox, model.nVox);
    const params = await store.loadMarker("ICVF");
    for (const key of ["hasModel", "m", "aTriu", "hyp", "meanKnots", "inscaler", "outscaler", "beMap", "covRanges"]) {
      assert.deepEqual(params[key], model.params.ICVF[key], key);
    }
    assert.deepEqual(params.beLevelColumns, ["Sex=0", "Sex=1"]);
  }));

test("the generated store is sharded and compressed like the real store", () =>
  withStore(null, async (dir) => {
    const root = JSON.parse(await readFile(join(dir, "zarr.json"), "utf8"));
    const meta = root.consolidated_metadata.metadata["FA/A_triu"];
    const sharding = meta.codecs[0];
    assert.equal(sharding.name, "sharding_indexed");
    assert.deepEqual(sharding.configuration.codecs.map((c) => c.name), ["bytes", "blosc"]);
    assert.deepEqual(sharding.configuration.index_codecs.map((c) => c.name), ["bytes", "crc32c"]);
    const shards = await readdir(join(dir, "FA", "A_triu", "c"));
    assert.equal(shards.length, Math.ceil(model.nVox / SHARD_VOXELS));
    assert.ok(model.nVox % SHARD_VOXELS !== 0, "the last shard is partial");
  }));

for (const [label, change, message] of [
  ["another warp", (attrs) => (attrs.regression.warp_name = "warpaffine"), /regression.warp_name is "warpaffine"/],
  ["an extra covariate", (attrs) => (attrs.covariates = ["Age", "prob_csf"]), /covariates are \["Age","prob_csf"\]/],
  ["a site batch effect", (attrs) => (attrs.batch_effects = { Sex: ["0", "1"], Scanner: ["a"] }), /batch effects are \["Sex","Scanner"\]/],
  ["a linear term in the basis", (attrs) => (attrs.basis_mean.include_linear = true), /basis_mean.include_linear/],
]) {
  test(`a marker fitted with ${label} is refused`, () =>
    withStore(
      (dir) => editJson(join(dir, "zarr.json"), (root) => change(root.consolidated_metadata.metadata.FA.attributes)),
      async (dir) => {
        const store = await openModelStore({ kind: "directory", handle: folderDirectory(dir) });
        await assert.rejects(store.loadMarker("FA"), message);
        await store.loadMarker("MD");
      },
    ));
}

test("optional options may be absent (PCNtoolkit's default), required ones may not", () =>
  withStore(
    (dir) => editJson(join(dir, "zarr.json"), (root) => {
      const fa = root.consolidated_metadata.metadata.FA.attributes;
      delete fa.y_transform;
      delete fa.regression.fixed_effect_var;
      delete root.consolidated_metadata.metadata.MD.attributes.regression.warp_name;
    }),
    async (dir) => {
      const store = await openModelStore({ kind: "directory", handle: folderDirectory(dir) });
      await store.loadMarker("FA");
      await assert.rejects(store.loadMarker("MD"), /regression.warp_name is missing, expected "warpsinharcsinh"/);
    },
  ));

test("a be_map entry outside the design columns is refused", async () => {
  const broken = syntheticModel({ markers: ["FA"] });
  const modelled = broken.params.FA.hasModel.indexOf(1, 2);
  const unmodelled = broken.params.FA.hasModel.indexOf(0);
  broken.params.FA.beMap[2 * unmodelled] = -1; // the real stores' fill value: allowed
  broken.params.FA.beMap[2 * modelled + 1] = 2;
  const dir = await mkdtemp(join(tmpdir(), "nmzarr-"));
  try {
    await writeSyntheticStore(dir, broken);
    const store = await openModelStore({ kind: "directory", handle: folderDirectory(dir) });
    await assert.rejects(store.loadMarker("FA"), new RegExp(`FA/be_map: voxel ${modelled} names design column 2, outside 0–1`));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("an array with the wrong width per voxel is refused", () =>
  withStore(
    (dir) => editJson(join(dir, "zarr.json"), (root) => (root.consolidated_metadata.metadata["FA/be_map"].shape[1] = 3)),
    async (dir) => {
      const store = await openModelStore({ kind: "directory", handle: folderDirectory(dir) });
      await assert.rejects(store.loadMarker("FA"), /FA\/be_map: expected 2 values per voxel, found 3/);
    },
  ));
