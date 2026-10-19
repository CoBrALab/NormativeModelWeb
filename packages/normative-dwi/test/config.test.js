import assert from "node:assert/strict";
import test from "node:test";
import { clampingNotice, guessMarker, parseCovariates, parseMarker, trainingRange, zmapFileName } from "../src/config.js";

test("markers are recognised from common file names", () => {
  for (const [name, marker] of [
    ["maps_UKB_dti_FA.nii.gz", "FA"],
    ["maps_UKB_dti_MD.nii", "MD"],
    ["maps_UKB_noddi_ICVF.nii.gz", "ICVF"],
    ["sub-01_desc-NDI_map.nii.gz", "ICVF"],
    ["maps_UKB_noddi_ISOVF.nii.gz", "ISOVF"],
    ["sub-01_FWF.nii.gz", "ISOVF"],
    ["maps_UKB_noddi_OD.nii.gz", "OD"],
    ["sub-01_odi.nii.gz", "OD"],
  ]) assert.equal(guessMarker(name), marker, name);
});

test("ambiguous or unknown names are left for the user", () => {
  assert.equal(guessMarker("subject.nii.gz"), null);
  assert.equal(guessMarker("FA_and_MD.nii.gz"), null);
  assert.equal(guessMarker("FAST.nii.gz"), null);
});

test("z-map file names", () => {
  assert.equal(zmapFileName("ISOVF"), "ISOVF_zscore.nii.gz");
});

test("training range covers only scored voxels", () => {
  const ranges = Float64Array.from([45, 81, 10, 99, 46, 80]);
  assert.deepEqual(trainingRange(ranges, Uint8Array.from([1, 0, 1])), { min: 45, max: 81 });
  assert.equal(trainingRange(ranges, Uint8Array.from([0, 0, 0])), null);
});

test("clamping notice only when an age was clamped", () => {
  assert.equal(clampingNotice(60, 0, { min: 45, max: 81 }), null);
  assert.equal(clampingNotice(38, 1000, { min: 45.2, max: 81.4 }), "Age 38 is outside the model's 45–81 range; scored at the nearest age");
});

test("covariates need a plausible age and a sex", () => {
  assert.deepEqual(parseCovariates("63.5", "female"), { age: 63.5, sex: "female" });
  assert.deepEqual(parseCovariates(" 72 ", "male"), { age: 72, sex: "male" });
  // The whole text must be a number: a comma or a unit is a typo, not 66 or 72.
  for (const [age, sex] of [["", "male"], [" ", "male"], ["undefined", "male"], ["0", "male"], ["-4", "female"], ["130", "male"], ["66,5", "female"], ["72y", "female"], ["60", ""], ["60", "undefined"], ["60", "F"]]) {
    assert.equal(parseCovariates(age, sex), null, `${age}/${sex}`);
  }
});

test("a marker is parsed from its name or a file-name alias, in any case", () => {
  for (const [name, marker] of [["FA", "FA"], ["md", "MD"], ["NDI", "ICVF"], ["fwf", "ISOVF"], ["FISO", "ISOVF"], ["odi", "OD"], [" OD ", "OD"]]) assert.equal(parseMarker(name), marker, name);
  for (const name of ["", "XY", "FA_MD", "T1"]) assert.equal(parseMarker(name), null, name);
});
