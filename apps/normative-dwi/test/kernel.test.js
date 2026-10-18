// CPU float64 kernel against PCNtoolkit (docs/adr/0001: |Δz| < 1e-6).
import assert from "node:assert/strict";
import test from "node:test";
import { zmapCpu } from "../src/predictor/kernel.ts";
import { FIXTURE_MARKERS, fixtureParams, loadFixture, sexOf } from "./fixtures.js";

const TOLERANCE = 1e-6;

for (const marker of FIXTURE_MARKERS) {
  test(`${marker}: CPU z matches PCNtoolkit at every fixture case`, () => {
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

test("one call scores every voxel for the same subject", () => {
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

test("missing values and unmodelled voxels are not scored", () => {
  const fixture = loadFixture("FA");
  const params = fixtureParams(fixture, fixture.voxels.slice(0, 4));
  params.hasModel[3] = 0;
  const { z, valid } = zmapCpu(params, { age: 60, sex: "female" }, [0, Number.NaN, Number.POSITIVE_INFINITY, 0.4]);
  assert.deepEqual([...valid], [0, 0, 0, 0]);
  assert.deepEqual([...z], [0, 0, 0, 0]);
});

test("a value count that does not match the store is rejected", () => {
  const fixture = loadFixture("FA");
  const params = fixtureParams(fixture, fixture.voxels.slice(0, 2));
  assert.throws(() => zmapCpu(params, { age: 60, sex: "female" }, [0.4]), /Expected 2 values/);
});
