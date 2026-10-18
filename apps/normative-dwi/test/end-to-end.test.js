// Whole-brain check on a real subject: NIfTI in → z-map NIfTI out, against
// tools/make_reference_zmap.py. Local data only; skipped unless both
//   NMZARR_STORE=<store>  and  NORMATIVE_E2E_DIR=<make_reference_zmap.py --out>
// are set.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { gunzipSync } from "fflate";
import { MARKERS } from "../src/model.ts";
import { openModelStore } from "../src/model-store/index.ts";
import { zmapCpu } from "../src/predictor/kernel.ts";
import { gatherToStore, gridMapping, loadSubjectMap, scatterToNifti } from "../src/subject-image/index.ts";
import { fsDirectory } from "./fs-directory.js";

const STORE = process.env.NMZARR_STORE;
const DIR = process.env.NORMATIVE_E2E_DIR;
const skip = STORE && DIR ? false : "set NMZARR_STORE and NORMATIVE_E2E_DIR";

for (const marker of MARKERS) {
  test(`${marker}: whole-brain z-map matches the reference`, { skip, timeout: 300000 }, async (t) => {
    const input = join(DIR, `input_${marker}.nii.gz`);
    if (!existsSync(input)) return t.skip(`no ${input}`);
    const { age, sex } = JSON.parse(readFileSync(join(DIR, "reference.json"), "utf8"));
    const store = await openModelStore({ kind: "directory", handle: fsDirectory(STORE) });
    const params = await store.loadMarker(marker);
    const map = await loadSubjectMap(`input_${marker}.nii.gz`, readFileSync(input));
    const mapping = gridMapping(map, store.grid);
    const { z, valid } = zmapCpu(params, { age, sex }, gatherToStore(map, mapping));
    const out = await loadSubjectMap("out", gunzipSync(scatterToNifti(z, valid, mapping, map)));
    const ref = await loadSubjectMap("ref", readFileSync(join(DIR, `reference_${marker}.nii.gz`)));
    assert.equal(out.data.length, ref.data.length);
    let worst = 0, scored = 0, zeroMismatch = 0;
    for (let i = 0; i < ref.data.length; i++) {
      if ((ref.data[i] === 0) !== (out.data[i] === 0)) zeroMismatch++;
      if (ref.data[i] !== 0) scored++;
      worst = Math.max(worst, Math.abs(out.data[i] - ref.data[i]) / Math.max(1, Math.abs(ref.data[i])));
    }
    assert.equal(zeroMismatch, 0);
    assert.ok(scored > 200000, `only ${scored} voxels scored`);
    assert.ok(worst < 1e-6, `max relative |Δz| ${worst}`);
  });
}
