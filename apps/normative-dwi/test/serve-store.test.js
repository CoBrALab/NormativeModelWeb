// The local store server used by the e2e specs and for development.
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { serveStore, storePath } from "../e2e/serve-store.mjs";

function get(port, path, headers = {}) {
  return new Promise((done, fail) => {
    const req = request({ host: "127.0.0.1", port, path, headers }, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => done({ status: res.statusCode, body: Buffer.concat(chunks).toString() }));
    });
    req.on("error", fail);
    req.end();
  });
}

test("serves files under a root given with or without a trailing slash, with ranges", async () => {
  const dir = await mkdtemp(join(tmpdir(), "serve-store-"));
  try {
    await writeFile(join(dir, "zarr.json"), "0123456789");
    for (const root of [dir, `${dir}/`]) {
      const server = await serveStore(root);
      const { port } = server.address();
      try {
        assert.deepEqual(await get(port, "/zarr.json"), { status: 200, body: "0123456789" });
        assert.deepEqual(await get(port, "/zarr.json", { Range: "bytes=2-4" }), { status: 206, body: "234" });
        assert.deepEqual(await get(port, "/zarr.json", { Range: "bytes=-3" }), { status: 206, body: "789" });
        assert.equal((await get(port, "/missing")).status, 404);
      } finally {
        server.close();
      }
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("malformed percent-encoding is a 400, not a crash", async () => {
  const dir = await mkdtemp(join(tmpdir(), "serve-store-"));
  const server = await serveStore(dir);
  try {
    assert.equal((await get(server.address().port, "/%E0%A4%A")).status, 400);
    assert.equal((await get(server.address().port, "/")).status, 404, "still serving afterwards");
  } finally {
    server.close();
    await rm(dir, { recursive: true, force: true });
  }
});

test("paths cannot escape the store", () => {
  assert.equal(storePath("/srv/store", "/../etc/passwd"), null);
  assert.equal(storePath("/srv/store", "/a/../../store2/x"), null);
  assert.equal(storePath("/srv/store", "/FA/m/c/0/0"), "/srv/store/FA/m/c/0/0");
});
