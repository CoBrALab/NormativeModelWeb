// Predictor adapters without a GPU: CPU blocks, cancellation and the GPU→CPU fallback.
import assert from "node:assert/strict";
import test from "node:test";
import { cpuPredictor, gpuPredictor } from "../src/predictor/index.ts";
import { zmapCpu } from "../src/predictor/kernel.ts";
import { syntheticMap, syntheticModel, syntheticParams } from "./synthetic-store.js";

const model = syntheticModel();
const params = syntheticParams(model, "FA");
const { y } = syntheticMap(model, "FA");
const subject = { age: 63, sex: "female" };

test("the CPU predictor scores in blocks and matches the synchronous kernel", async () => {
  const expected = zmapCpu(params, subject, y);
  const got = await cpuPredictor().zmap(params, subject, y);
  assert.equal(got.kind, "cpu");
  assert.deepEqual(got.z, expected.z);
  assert.deepEqual(got.valid, expected.valid);
});

test("an aborted signal cancels CPU scoring", async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(cpuPredictor().zmap(params, subject, y, controller.signal), { name: "AbortError" });
});

function fakeGpu({ fail = null, lost = null } = {}) {
  let disposed = 0;
  return {
    gpu: {
      calls: 0,
      lostReason: () => lost,
      dispose: () => disposed++,
      async zmap() {
        this.calls++;
        if (fail) throw new Error(fail);
        return { z: new Float64Array(params.nVox), valid: new Uint8Array(params.nVox), clamped: 0 };
      },
    },
    disposed: () => disposed,
  };
}

test("a failing GPU hands this and later markers to the CPU, once", async () => {
  const fake = fakeGpu({ fail: "device hung" });
  const reasons = [];
  const predictor = gpuPredictor(fake.gpu, (reason) => reasons.push(reason));
  const first = await predictor.zmap(params, subject, y);
  const second = await predictor.zmap(params, subject, y);
  assert.equal(first.kind, "cpu");
  assert.equal(second.kind, "cpu");
  assert.deepEqual(first.z, zmapCpu(params, subject, y).z);
  assert.deepEqual(reasons, ["device hung"]);
  assert.equal(fake.gpu.calls, 1);
  assert.equal(fake.disposed(), 1);
  assert.equal(predictor.kind, "cpu");
});

test("a lost device is not used at all", async () => {
  const fake = fakeGpu({ lost: "lost: destroyed" });
  const reasons = [];
  const predictor = gpuPredictor(fake.gpu, (reason) => reasons.push(reason));
  assert.equal(predictor.kind, "cpu");
  const result = await predictor.zmap(params, subject, y);
  assert.equal(result.kind, "cpu");
  assert.equal(fake.gpu.calls, 0);
  assert.deepEqual(reasons, ["lost: destroyed"]);
});
