// ModelStore on a generated store: reading, the design and layout checks, and
// the response cache's rules. Needs no data.
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { openModelStore } from "../src/model-store/index.ts";
import { cachingFetch, clearModelCache, isComplete, isPinnedUrl } from "../src/model-store/persistent-cache.ts";
import { fsDirectory } from "./fs-directory.js";
import { SYNTHETIC_FAMILY, syntheticModel, writeSyntheticStore } from "./synthetic-store.js";

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
    const store = await openModelStore({ kind: "directory", handle: fsDirectory(dir) });
    assert.equal(store.family, SYNTHETIC_FAMILY);
    assert.deepEqual(store.grid.shape, model.shape);
    assert.equal(store.grid.nVox, model.nVox);
    const params = await store.loadMarker("ICVF");
    for (const key of ["hasModel", "m", "aTriu", "hyp", "meanKnots", "inscaler", "outscaler", "beMap", "covRanges"]) {
      assert.deepEqual(params[key], model.params.ICVF[key], key);
    }
    assert.deepEqual(params.beLevelColumns, ["Sex=0", "Sex=1"]);
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
        const store = await openModelStore({ kind: "directory", handle: fsDirectory(dir) });
        await assert.rejects(store.loadMarker("FA"), message);
        await store.loadMarker("MD");
      },
    ));
}

test("an array with the wrong width per voxel is refused", () =>
  withStore(
    (dir) => editJson(join(dir, "zarr.json"), (root) => (root.consolidated_metadata.metadata["FA/be_map"].shape[1] = 3)),
    async (dir) => {
      const store = await openModelStore({ kind: "directory", handle: fsDirectory(dir) });
      await assert.rejects(store.loadMarker("FA"), /FA\/be_map: expected 2 values per voxel, found 3/);
    },
  ));

test("only Hugging Face URLs pinned to a commit count as immutable", () => {
  assert.ok(isPinnedUrl(`https://huggingface.co/datasets/neurodeskorg/webapps/resolve/${"a".repeat(40)}/normative-dwi/store/zarr.json`));
  assert.ok(!isPinnedUrl("https://huggingface.co/datasets/neurodeskorg/webapps/resolve/main/normative-dwi/store/zarr.json"));
  assert.ok(!isPinnedUrl("http://127.0.0.1:8765/zarr.json"));
});

test("only complete responses are cacheable", () => {
  const ranged = (range) => new Request("https://x/a", { headers: { Range: range } });
  const body = (n) => new ArrayBuffer(n);
  assert.ok(isComplete(new Request("https://x/a"), new Response(null, { status: 200 }), body(5)));
  assert.ok(isComplete(ranged("bytes=10-19"), new Response(null, { status: 206 }), body(10)));
  assert.ok(!isComplete(ranged("bytes=10-19"), new Response(null, { status: 200 }), body(100)), "a server that ignored Range");
  assert.ok(!isComplete(ranged("bytes=10-19"), new Response(null, { status: 206 }), body(9)), "a short body");
  assert.ok(isComplete(ranged("bytes=-16"), new Response(null, { status: 206 }), body(16)));
});

function fakeCaches() {
  const stores = new Map();
  return {
    stores,
    async open(name) {
      if (!stores.has(name)) stores.set(name, new Map());
      const entries = stores.get(name);
      return { match: async (key) => entries.get(key)?.clone(), put: async (key, response) => void entries.set(key, response) };
    },
    keys: async () => [...stores.keys()],
    delete: async (name) => stores.delete(name),
  };
}

test("pinned responses are served from the cache, others always go to the network, and the cache clears", async (t) => {
  const pinned = `https://hf.example/resolve/${"b".repeat(40)}/store/zarr.json`;
  let calls = 0;
  t.mock.method(globalThis, "fetch", async () => {
    calls++;
    return new Response("{}", { status: 200 });
  });
  const storage = fakeCaches();
  const fetchVia = cachingFetch(storage);
  await fetchVia(new Request(pinned));
  await fetchVia(new Request(pinned));
  assert.equal(calls, 1);
  await fetchVia(new Request("http://127.0.0.1:8765/zarr.json"));
  await fetchVia(new Request("http://127.0.0.1:8765/zarr.json"));
  assert.equal(calls, 3);
  assert.equal(await clearModelCache(storage), true);
  await cachingFetch(storage)(new Request(pinned));
  assert.equal(calls, 4);
});
