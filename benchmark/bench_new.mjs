// Time the nmzarr store + the app's float64 CPU kernel on one marker.
//
// Uses the app's own code: openModelStore (zarrita, sharding, blosc/zstd) on a
// local folder, then zmapCpu, the kernel the browser runs. The whole marker is
// read and every modelled voxel scored; nothing is sampled or extrapolated.
//
// Usage: node benchmark/bench_new.mjs --store STORE.zarr --manifest DIR/manifest.json --out results/new.json
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { openModelStore } from "../apps/normative-dwi/src/model-store/index.ts";
import { zmapCpu } from "../apps/normative-dwi/src/predictor/kernel.ts";
import { fsDirectory } from "../apps/normative-dwi/test/fs-directory.js";
import * as subjects from "./subjects.mjs";

const { values: args } = parseArgs({
  options: {
    store: { type: "string" },
    manifest: { type: "string" },
    out: { type: "string" },
    subjects: { type: "string", default: "1,100" },
    repeats: { type: "string", default: "5" },
  },
});
const manifest = JSON.parse(readFileSync(args.manifest, "utf8"));
const cohorts = args.subjects.split(",").map(Number);
const repeats = Number(args.repeats);
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

let t = performance.now();
const store = await openModelStore({ kind: "directory", handle: fsDirectory(args.store) });
const openMs = performance.now() - t;

const loadMs = [];
let params;
for (let i = 0; i < repeats; i++) {
  t = performance.now();
  params = await store.loadMarker(manifest.marker);
  loadMs.push(performance.now() - t);
}

function subjectMap(s) {
  const y = new Float64Array(params.nVox);
  for (let v = 0; v < params.nVox; v++) {
    if (params.hasModel[v]) y[v] = subjects.response(v, s, params.outscaler[4 * v], params.outscaler[4 * v + 1]);
  }
  return y;
}

const configs = {};
let z0;
for (const n of cohorts) {
  const maps = Array.from({ length: n }, (_, s) => subjectMap(s));
  const runs = [];
  let scored = 0;
  for (let r = 0; r < repeats; r++) {
    t = performance.now();
    for (let s = 0; s < n; s++) {
      const result = zmapCpu(params, { age: subjects.age(s), sex: subjects.sex(s) }, maps[s]);
      if (s === 0 && r === 0) {
        z0 = result;
        scored = result.valid.reduce((a, b) => a + b, 0);
      }
    }
    runs.push(performance.now() - t);
  }
  const scoreMs = median(runs);
  configs[`zscores/${n}`] = {
    subjects: n,
    score_ms_runs: runs,
    score_ms: scoreMs,
    us_per_voxel_per_subject: (1e3 * scoreMs) / (n * scored),
    marker_s: (median(loadMs) + scoreMs) / 1e3,
    scored_subject0: scored,
  };
}

const zSubject0 = {};
for (const v of manifest.sample) zSubject0[String(v)] = z0.z[v];

const doc = {
  implementation: "nmzarr store + app CPU kernel (float64)",
  node: process.version,
  store: args.store,
  n_vox: params.nVox,
  n_models: params.hasModel.reduce((a, b) => a + b, 0),
  open_ms: openMs,
  load_ms_runs: loadMs,
  load_ms: median(loadMs),
  us_per_voxel_read: (1e3 * median(loadMs)) / params.hasModel.reduce((a, b) => a + b, 0),
  configs,
  z_subject0: zSubject0,
};
mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, JSON.stringify(doc, null, 1) + "\n");
for (const [key, c] of Object.entries(configs)) {
  console.log(`${key}: load ${doc.load_ms.toFixed(0)} ms + score ${c.score_ms.toFixed(0)} ms (${c.us_per_voxel_per_subject.toFixed(3)} µs/voxel/subject)`);
}
