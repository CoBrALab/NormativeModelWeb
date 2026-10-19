// The model store response cache's rules. Needs no data.
import assert from "node:assert/strict";
import test from "node:test";
import { CACHE_NAME, cacheKey, isComplete, isPinnedUrl, pruneOldCaches } from "../src/model-store/persistent-cache.ts";

test("only Hugging Face URLs pinned to a commit count as immutable", () => {
  const pinned = `https://huggingface.co/datasets/neurodeskorg/webapps/resolve/${"a".repeat(40)}/normative-dwi/store/zarr.json`;
  assert.ok(isPinnedUrl(pinned));
  assert.ok(isPinnedUrl(pinned.replace("huggingface.co", "hf.co")));
  assert.ok(!isPinnedUrl(pinned.replace(/resolve\/a+/, "resolve/main")));
  assert.ok(!isPinnedUrl(pinned.replace("huggingface.co", "evil.example")), "another host with the same path");
  assert.ok(!isPinnedUrl(pinned.replace("https:", "http:")));
  assert.ok(!isPinnedUrl("http://127.0.0.1:8765/zarr.json"));
});

test("only complete responses are cacheable", () => {
  const ranged = (range) => new Request("https://x/a", { headers: { Range: range } });
  const reply = (status, contentRange) => new Response(null, { status, headers: contentRange ? { "Content-Range": contentRange } : {} });
  const body = (n) => new ArrayBuffer(n);
  assert.ok(isComplete(new Request("https://x/a"), reply(200), body(5)));
  assert.ok(isComplete(ranged("bytes=10-19"), reply(206, "bytes 10-19/100"), body(10)));
  assert.ok(isComplete(ranged("bytes=90-119"), reply(206, "bytes 90-99/100"), body(10)), "cut at the end of the file");
  assert.ok(!isComplete(ranged("bytes=10-19"), reply(200), body(100)), "a server that ignored Range");
  assert.ok(!isComplete(ranged("bytes=10-19"), reply(206, "bytes 10-19/100"), body(9)), "a short body");
  assert.ok(!isComplete(ranged("bytes=10-19"), reply(206, "bytes 20-29/100"), body(10)), "another range");
  assert.ok(!isComplete(ranged("bytes=10-19"), reply(206), body(10)), "no Content-Range");
  assert.ok(isComplete(ranged("bytes=-16"), reply(206, "bytes 84-99/100"), body(16)));
  assert.ok(!isComplete(ranged("bytes=-16"), reply(206, "bytes 80-95/100"), body(16)), "a suffix that is not the end");
});

test("cache keys keep ranges apart in the query string, not a fragment", () => {
  const key = (range) => cacheKey(new Request("https://hf.co/a/resolve/x/c/0/0", { headers: { Range: range } }));
  assert.notEqual(key("bytes=0-3"), key("bytes=4-7"));
  assert.match(key("bytes=0-3"), /\?nd-range=0-3$/);
  assert.ok(!key("bytes=0-3").includes("#"));
});

test("stale caches of earlier versions are pruned, the current one kept", async () => {
  const names = new Set(["normative-dwi-model-store-v1", "normative-dwi-model-store-v2", CACHE_NAME, "other-app"]);
  const storage = { keys: async () => [...names], delete: async (name) => names.delete(name) };
  await pruneOldCaches(storage);
  assert.deepEqual([...names].sort(), [CACHE_NAME, "other-app"].sort());
});
