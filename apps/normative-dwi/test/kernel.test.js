// CPU float64 kernel. Parity with PCNtoolkit (docs/adr/0001: |Δz| < 1e-6) needs
// the local fixtures from tools/make_fixtures.py, which hold UK Biobank-derived
// parameters and are never committed; the other tests use made-up parameters.
import assert from "node:assert/strict";
import test from "node:test";
import { bsplineBasis, zmapCpu } from "../src/predictor/kernel.ts";
import { FIXTURE_MARKERS, fixtureParams, fixturesAvailable, loadFixture, sexOf } from "./fixtures.js";
import { syntheticMap, syntheticModel, syntheticParams } from "./synthetic-store.js";

const TOLERANCE = 1e-6;
const skip = fixturesAvailable() ? false : "no local fixtures (run tools/make_fixtures.py)";

for (const marker of FIXTURE_MARKERS) {
  test(`${marker}: CPU z matches PCNtoolkit at every fixture case`, { skip }, () => {
    const fixture = loadFixture(marker);
    let worst = 0;
    let cases = 0;
    for (const record of fixture.voxels) {
      const params = fixtureParams(fixture, [record]);
      for (const c of record.cases) {
        const { z, valid, clamped } = zmapCpu(params, { age: c.age, sex: sexOf(c.sex) }, [c.y]);
        assert.equal(valid[0], 1);
        assert.equal(clamped, c.clamped ? 1 : 0, `voxel ${record.voxel} age ${c.age}`);
        worst = Math.max(worst, Math.abs(z[0] - c.z));
        cases++;
      }
    }
    assert.ok(worst < TOLERANCE, `max |Δz| ${worst} over ${cases} cases`);
  });
}

test("one call scores every voxel for the same subject", { skip }, () => {
  const fixture = loadFixture("FA");
  const params = fixtureParams(fixture);
  const pick = (r) => r.cases.find((c) => c.sex === "1" && !c.clamped && c.age === r.cases[2 * 2].age);
  const cases = fixture.voxels.map(pick);
  // Same age for all voxels is not guaranteed by the fixture; score each voxel at its own case age.
  for (const [v, c] of cases.entries()) {
    const y = new Float64Array(params.nVox);
    y[v] = c.y;
    const { z, valid } = zmapCpu(params, { age: c.age, sex: "male" }, y);
    assert.equal(valid.reduce((a, b) => a + b, 0), 1);
    assert.ok(Math.abs(z[v] - c.z) < TOLERANCE);
  }
});

const model = syntheticModel({ markers: ["FA"] });
const firstModelled = (n) => {
  const params = syntheticParams(model, "FA");
  const picks = [...params.hasModel.keys()].filter((v) => params.hasModel[v]).slice(0, n);
  return { params: { ...params, nVox: n, ...Object.fromEntries(["hasModel", "m", "aTriu", "hyp", "meanKnots", "inscaler", "outscaler", "beMap", "covRanges"].map((key) => {
    const width = params[key].length / params.nVox;
    return [key, params[key].constructor.from(picks.flatMap((v) => [...params[key].subarray(v * width, (v + 1) * width)]))];
  })) }, picks };
};

test("missing values and unmodelled voxels are not scored", () => {
  const { params } = firstModelled(4);
  params.hasModel[3] = 0;
  const { z, valid } = zmapCpu(params, { age: 60, sex: "female" }, [0, Number.NaN, Number.POSITIVE_INFINITY, 0.4]);
  assert.deepEqual([...valid], [0, 0, 0, 0]);
  assert.deepEqual([...z], [0, 0, 0, 0]);
});

test("a value count that does not match the store is rejected", () => {
  const { params } = firstModelled(2);
  assert.throws(() => zmapCpu(params, { age: 60, sex: "female" }, [0.4]), /Expected 2 values/);
});

test("a voxel whose precision is not positive definite is left unscored", () => {
  const { params } = firstModelled(2);
  for (let t = 0; t < 45; t++) params.aTriu[t] = 0; // voxel 0: A = 0
  params.aTriu[0] = -1;
  const { z, valid } = zmapCpu(params, { age: 60, sex: "female" }, [0.4, 0.4]);
  assert.deepEqual([...valid], [0, 1]);
  assert.equal(z[0], 0);
  assert.ok(Number.isFinite(z[1]));
});

test("the synthetic store's maps score finite z wherever there is a model and a value", () => {
  const params = syntheticParams(model, "FA");
  const { y } = syntheticMap(model, "FA");
  const { valid } = zmapCpu(params, { age: 70, sex: "male" }, y);
  for (let v = 0; v < params.nVox; v++) assert.equal(valid[v], params.hasModel[v] && y[v] !== 0 ? 1 : 0);
});

// scipy BSpline.design_matrix(x, knots, 3, extrapolate=True) on made-up knots
// spanning [-3.36, 3.36], including the extended end pieces outside it.
const KNOTS = [-3.36, -3.36, -3.36, -3.36, -1.12, 1.1199999999999997, 3.36, 3.36, 3.36, 3.36];
const SCIPY = [
  [-5.0, [5.196980912900876, -5.295370296556124, 1.1637981922679788, -0.06540880861273082, 0.0, 0.0]],
  [-3.36, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  [-1.0, [0.0, 0.21193541590743437, 0.5932285023384353, 0.1947976456207483, 3.8436133381924296e-05, 0.0]],
  [0.0, [0.0, 0.031249999999999986, 0.46875, 0.46875, 0.03125000000000001, 0.0]],
  [0.5, [0.0, 0.0053011613406523245, 0.3309953131643282, 0.5691362238254677, 0.09456730166955177, 0.0]],
  [3.359999999, [0.0, 0.0, 1.4828758843589808e-29, 2.9894775347016436e-19, 1.3392858242018678e-09, 0.9999999986607143]],
  [3.36, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
  [5.0, [0.0, 0.0, -0.06540880861273081, 1.1637981922679788, -5.295370296556122, 5.196980912900874]],
];

test("the cubic B-spline basis matches scipy, extrapolation included", () => {
  const knots = Float64Array.from(KNOTS);
  const out = new Float64Array(4), left = new Float64Array(4), right = new Float64Array(4);
  for (const [x, expected] of SCIPY) {
    const first = bsplineBasis(knots, 0, 10, x, out, left, right);
    const full = new Array(6).fill(0);
    for (let r = 0; r < 4; r++) full[first + r] = out[r];
    for (let i = 0; i < 6; i++) assert.ok(Math.abs(full[i] - expected[i]) < 1e-12, `x=${x} basis ${i}: ${full[i]} vs ${expected[i]}`);
  }
});
