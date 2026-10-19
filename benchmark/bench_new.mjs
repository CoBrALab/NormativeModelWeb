// Time the nmzarr store + the app's float64 CPU predictor on one marker.
//
// Uses the app's own code: openModelStore (zarrita, sharding, blosc/zstd) on a
// local folder, then cpuPredictor().zmap, which is what the browser calls
// (zmapCpuAsync: the same scorer, yielding to the event loop after every
// 16,384-voxel block so that scoring can be cancelled). The whole marker is read
// and every modelled voxel scored; nothing is sampled or extrapolated.
//
// marker_s = store open + first read + scoring (the first of the repeated runs).
// marker_s_warm uses the medians of the repeats instead.
//
// Usage: node benchmark/bench_new.mjs --store STORE.zarr --manifest DIR/manifest.json --out results/new.json
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { basename, dirname } from "node:path";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { openModelStore } from "../apps/normative-dwi/src/model-store/index.ts";
import { cpuPredictor } from "../apps/normative-dwi/src/predictor/index.ts";
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
for (const name of ["store", "manifest", "out"]) {
  if (!args[name]) {
    console.error(`Error: --${name} is required\nUsage: node benchmark/bench_new.mjs --store STORE.zarr --manifest MANIFEST.json --out OUT.json [--subjects 1,100] [--repeats 5]`);
    process.exit(1);
  }
}
const manifest = JSON.parse(readFileSync(args.manifest, "utf8"));
const cohorts = args.subjects.split(",").map(Number);
const repeats = Number(args.repeats);
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const predictor = cpuPredictor();

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
const nModels = params.hasModel.reduce((a, b) => a + b, 0);

function subjectMap(s) {
  const y = new Float64Array(params.nVox);
  for (let v = 0; v < params.nVox; v++) {
    if (params.hasModel[v]) y[v] = subjects.response(v, s, params.outscaler[4 * v], params.outscaler[4 * v + 1]);
  }
  return y;
}

const configs = {};
const zSubjects = {}; // voxel -> [z of subject 0 (female), z of subject 1 (male)] at the timed voxels
for (const n of cohorts) {
  const maps = Array.from({ length: n }, (_, s) => subjectMap(s));
  const runs = [];
  let scored = 0;
  for (let r = 0; r < repeats; r++) {
    t = performance.now();
    for (let s = 0; s < n; s++) {
      const result = await predictor.zmap(params, { age: subjects.age(s), sex: subjects.sex(s) }, maps[s]);
      if (r === 0 && s === 0) scored = result.valid.reduce((a, b) => a + b, 0);
      if (r === 0 && s < 2 && n === Math.max(...cohorts)) {
        for (const v of manifest.sample) (zSubjects[String(v)] ??= [])[s] = result.z[v];
      }
    }
    runs.push(performance.now() - t);
  }
  configs[`zscores/${n}`] = {
    subjects: n,
    score_ms_runs: runs,
    score_ms_first: runs[0],
    score_ms: median(runs),
    us_per_voxel_per_subject: (1e3 * median(runs)) / (n * scored),
    marker_s: (openMs + loadMs[0] + runs[0]) / 1e3,
    marker_s_warm: (median(loadMs) + median(runs)) / 1e3,
    scored_subject0: scored,
  };
}

const doc = {
  implementation: "nmzarr store + app CPU predictor (float64, zmapCpuAsync)",
  node: process.version,
  store: basename(args.store),
  n_vox: params.nVox,
  n_models: nModels,
  open_ms: openMs,
  load_ms_runs: loadMs,
  load_ms_first: loadMs[0],
  load_ms: median(loadMs),
  us_per_voxel_read: (1e3 * median(loadMs)) / nModels,
  configs,
  z_subjects01: zSubjects,
};
mkdirSync(dirname(args.out), { recursive: true });
writeFileSync(args.out, JSON.stringify(doc, null, 1) + "\n");
for (const [key, c] of Object.entries(configs)) {
  console.log(`${key}: open ${openMs.toFixed(0)} + first read ${loadMs[0].toFixed(0)} + first score ${c.score_ms_first.toFixed(0)} ms = ${(c.marker_s * 1e3).toFixed(0)} ms; warm ${(c.marker_s_warm * 1e3).toFixed(0)} ms (${c.us_per_voxel_per_subject.toFixed(3)} µs/voxel/subject)`);
}
