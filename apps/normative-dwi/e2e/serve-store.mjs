#!/usr/bin/env node
// Serve a local nmzarr store the way a public host must: HTTP Range requests
// (including suffix ranges, which sharded Zarr uses for shard indexes) and CORS.
// `python -m http.server` ignores Range and breaks shard reads.
//   node apps/normative-dwi/e2e/serve-store.mjs <store-dir> [port]
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { isAbsolute, relative, resolve } from "node:path";

/** The file a request path names inside `base`, or null when it escapes it. */
export function storePath(base, pathname) {
  const path = resolve(base, `.${pathname}`);
  const rel = relative(base, path);
  return rel === "" || (!rel.startsWith("..") && !isAbsolute(rel)) ? path : null;
}

export function serveStore(root, port = 0) {
  const base = resolve(root);
  const server = createServer((request, response) => {
    response.setHeader("Access-Control-Allow-Origin", "*");
    response.setHeader("Access-Control-Allow-Headers", "Range");
    response.setHeader("Access-Control-Expose-Headers", "Content-Length, Content-Range");
    response.setHeader("Cross-Origin-Resource-Policy", "cross-origin");
    if (request.method === "OPTIONS") return response.writeHead(204).end();
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(request.url, "http://x").pathname);
    } catch {
      return response.writeHead(400).end();
    }
    const path = storePath(base, pathname);
    const stat = path ? statSync(path, { throwIfNoEntry: false }) : undefined;
    if (!stat?.isFile()) return response.writeHead(404).end();
    const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range ?? "");
    let start = 0;
    let end = stat.size - 1;
    if (range) {
      if (range[1] === "") start = Math.max(0, stat.size - Number(range[2]));
      else {
        start = Number(range[1]);
        if (range[2] !== "") end = Math.min(end, Number(range[2]));
      }
      if (start > end) return response.writeHead(416, { "Content-Range": `bytes */${stat.size}` }).end();
      response.writeHead(206, { "Content-Range": `bytes ${start}-${end}/${stat.size}`, "Content-Length": end - start + 1, "Accept-Ranges": "bytes" });
    } else {
      response.writeHead(200, { "Content-Length": stat.size, "Accept-Ranges": "bytes" });
    }
    if (request.method === "HEAD") return response.end();
    createReadStream(path, { start, end }).pipe(response);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [root, port = "8765"] = process.argv.slice(2);
  if (!root) {
    console.error("usage: node e2e/serve-store.mjs <store-dir> [port]");
    process.exit(1);
  }
  const server = await serveStore(root, Number(port));
  console.log(`serving ${root} at http://127.0.0.1:${server.address().port}/`);
}
