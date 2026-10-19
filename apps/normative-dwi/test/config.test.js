import assert from "node:assert/strict";
import test from "node:test";
import { APP, parseZLimit } from "../src/config.ts";

test("app id is lowercase kebab-case and config is frozen", () => {
  assert.match(APP.id, /^[a-z][a-z0-9-]*$/);
  assert.ok(Object.isFrozen(APP));
});

test("the display limit is a |z| between 0.5 and 20", () => {
  for (const [text, limit] of [["3", 3], [" 2.5 ", 2.5], ["0.5", 0.5], ["20", 20], ["1e1", 10]]) assert.equal(parseZLimit(text), limit, text);
  for (const text of ["", " ", "0", "-3", "0.4", "21", "abc", "NaN", "Infinity"]) assert.equal(parseZLimit(text), null, text);
});
