// Keep fetched store bytes in Cache Storage so a second visit costs nothing.
//
// Only immutable URLs are cached: Hugging Face (huggingface.co, hf.co) resolve
// URLs pinned to a 40-hex commit. Anything else (a user's server,
// `resolve/main`) can change under the same URL, and mixing old and new chunks
// would give wrong z-scores with no error.
//
// The Cache API refuses 206 responses and ignores URL fragments, so a range is
// stored as a 200 response under a key with the range in its query string
// (`?nd-range=…`), and handed back to zarrita as a 206. Only complete answers
// are stored: a whole object with 200, or a range whose Content-Range matches
// the request and whose body has the matching length. Without Cache Storage
// (Node, insecure contexts, quota errors) requests pass straight through.

const CACHE_PREFIX = "normative-dwi-model-store";
export const CACHE_NAME = `${CACHE_PREFIX}-v3`;

const PINNED_HOSTS = new Set(["huggingface.co", "hf.co"]);
const PINNED_PATH = /\/resolve\/[0-9a-f]{40}\//;

export function isPinnedUrl(url: string): boolean {
  const { protocol, hostname, pathname } = new URL(url);
  return protocol === "https:" && PINNED_HOSTS.has(hostname) && PINNED_PATH.test(pathname);
}

type Wanted = { suffix: number } | { start: number; end: number };

function requested(range: string | null): Wanted | null {
  const match = range && /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match || (match[1] === "" && match[2] === "")) return null;
  if (match[1] === "") return { suffix: Number(match[2]) };
  return { start: Number(match[1]), end: Number(match[2]) };
}

/**
 * Whether a response is a complete answer to its request and so safe to cache:
 * 200 for a whole object; for a range, 206 with a Content-Range covering
 * exactly the requested bytes (cut at the end of the file) and that many bytes.
 */
export function isComplete(request: Request, response: Response, body: ArrayBuffer): boolean {
  const header = request.headers.get("Range");
  if (!header) return response.status === 200;
  const wanted = requested(header);
  const got = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("Content-Range") ?? "");
  if (response.status !== 206 || !wanted || !got) return false;
  const [start, end, total] = got.slice(1).map(Number);
  if (end - start + 1 !== body.byteLength || end >= total) return false;
  if ("suffix" in wanted) return end === total - 1 && body.byteLength === Math.min(wanted.suffix, total);
  return start === wanted.start && end === Math.min(wanted.end, total - 1);
}

/** The cache key: the URL, plus the range in the query (the Cache API drops fragments). */
export function cacheKey(request: Request): string {
  const range = request.headers.get("Range");
  if (!range) return request.url;
  const url = new URL(request.url);
  url.searchParams.set("nd-range", range.replace(/^bytes=/, ""));
  return url.href;
}

/** A fetch for zarrita's FetchStore that reads and fills the persistent cache. */
export function cachingFetch(storage: CacheStorage | undefined = (globalThis as { caches?: CacheStorage }).caches): (request: Request) => Promise<Response> {
  let opened: Promise<Cache | null> | null = null;
  const cache = () => (opened ??= storage ? storage.open(CACHE_NAME).catch(() => null) : Promise.resolve(null));
  return async (request) => {
    if (request.method !== "GET" || !isPinnedUrl(request.url)) return fetch(request);
    const c = await cache();
    if (!c) return fetch(request);
    const key = cacheKey(request);
    const ranged = request.headers.has("Range");
    const hit = await c.match(key).catch(() => undefined);
    if (hit) {
      if (!ranged) return hit;
      return new Response(await hit.arrayBuffer(), { status: 206, headers: { "Content-Range": hit.headers.get("X-Content-Range") ?? "" } });
    }
    const response = await fetch(request);
    if (!response.ok) return response;
    const body = await response.arrayBuffer();
    if (isComplete(request, response, body)) {
      const headers: Record<string, string> = { "Content-Type": "application/octet-stream" };
      if (ranged) headers["X-Content-Range"] = response.headers.get("Content-Range") ?? "";
      await c.put(key, new Response(body.slice(0), { status: 200, headers })).catch(() => undefined);
    }
    return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
}

/** Delete caches left by earlier versions of this app; the current one is kept. */
export async function pruneOldCaches(storage: CacheStorage | undefined = (globalThis as { caches?: CacheStorage }).caches): Promise<void> {
  if (!storage) return;
  const names = await storage.keys().catch(() => [] as string[]);
  await Promise.all(names.filter((name) => name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME).map((name) => storage.delete(name)));
}

/** Remove every cached model-store response; returns whether anything was cleared. */
export async function clearModelCache(storage: CacheStorage | undefined = (globalThis as { caches?: CacheStorage }).caches): Promise<boolean> {
  if (!storage) return false;
  const names = await storage.keys();
  const ours = names.filter((name) => name.startsWith(CACHE_PREFIX));
  await Promise.all(ours.map((name) => storage.delete(name)));
  return ours.length > 0;
}
