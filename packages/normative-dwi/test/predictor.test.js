// The CPU predictor: blocks and cancellation.
import assert from "node:assert/strict";
import test from "node:test";
import { cpuPredictor } from "../src/predictor/index.js";
import { zmapCpu } from "../src/predictor/kernel.js";
import { syntheticMap, syntheticModel, syntheticParams } from "../src/synthetic-store.js";

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

test("blocked CPU scoring matches in many blocks and can be cancelled between them", async () => {
  const { zmapCpuAsync } = await import("../src/predictor/kernel.js");
  const expected = zmapCpu(params, subject, y);
  const many = await zmapCpuAsync(params, subject, y, undefined, 97);
  assert.deepEqual(many.z, expected.z);
  // Abort after the first yield: the run must stop before finishing.
  const controller = new AbortController();
  const run = zmapCpuAsync(params, subject, y, controller.signal, 64);
  setTimeout(() => controller.abort(), 0);
  await assert.rejects(run, { name: "AbortError" });
});
