// Keep fetched store bytes in Cache Storage so a second visit costs nothing.
// Only immutable URLs are cached: a Hugging Face resolve URL pinned to a
// 40-hex commit. Anything else (a user's server, `resolve/main`) can change
// under the same URL, and mixing old and new chunks would give wrong z-scores
// with no error. Only complete responses are kept: 200 for a whole object,
// 206 of exactly the requested length for a range. Without Cache Storage
// (Node, insecure contexts, quota errors) requests pass straight through.

export const CACHE_NAME = "normative-dwi-model-store-v2";

const PINNED = /\/resolve\/[0-9a-f]{40}\//;

export function isPinnedUrl(url: string): boolean {
  return PINNED.test(new URL(url).pathname);
}

function requestedLength(range: string | null): { suffix: boolean; length: number } | null {
  const match = range && /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) return null;
  if (match[1] === "") return { suffix: true, length: Number(match[2]) };
  return { suffix: false, length: Number(match[2]) - Number(match[1]) + 1 };
}

/** Whether a response is a complete answer to its request and so safe to cache. */
export function isComplete(request: Request, response: Response, body: ArrayBuffer): boolean {
  const range = request.headers.get("Range");
  if (!range) return response.status === 200;
  const wanted = requestedLength(range);
  if (response.status !== 206 || !wanted) return false;
  return wanted.suffix ? body.byteLength <= wanted.length && body.byteLength > 0 : body.byteLength === wanted.length;
}

function cacheKey(request: Request): string {
  const range = request.headers.get("Range");
  return range ? `${request.url}#${encodeURIComponent(range)}` : request.url;
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
    const hit = await c.match(key).catch(() => undefined);
    if (hit) return hit;
    const response = await fetch(request);
    if (!response.ok) return response;
    const body = await response.arrayBuffer();
    const copy = () => new Response(body.slice(0), { status: response.status, statusText: response.statusText, headers: response.headers });
    if (isComplete(request, response, body)) await c.put(key, copy()).catch(() => undefined);
    return copy();
  };
}

/** Remove every cached model-store response; returns whether anything was cleared. */
export async function clearModelCache(storage: CacheStorage | undefined = (globalThis as { caches?: CacheStorage }).caches): Promise<boolean> {
  if (!storage) return false;
  const names = await storage.keys();
  const ours = names.filter((name) => name.startsWith("normative-dwi-model-store"));
  await Promise.all(ours.map((name) => storage.delete(name)));
  return ours.length > 0;
}
