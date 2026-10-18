// Keep fetched store bytes in Cache Storage so a second visit costs nothing.
// Store URLs are pinned (immutable), so entries never go stale; a new store
// version has a new URL and therefore new keys. Without Cache Storage (Node,
// insecure contexts, quota errors) reads simply pass through.
import type { AbsolutePath, AsyncReadable, RangeQuery } from "zarrita";

export const CACHE_NAME = "normative-dwi-model-store-v1";

function cacheKey(base: string, key: AbsolutePath, range?: RangeQuery): string {
  const suffix = !range ? "" : "suffixLength" in range ? `?suffix=${range.suffixLength}` : `?offset=${range.offset}&length=${range.length}`;
  return `https://normative-dwi.cache.invalid/${encodeURIComponent(base)}${key}${suffix}`;
}

export function persistentCache(store: AsyncReadable, base: string): AsyncReadable {
  const storage = (globalThis as { caches?: CacheStorage }).caches;
  if (!storage) return store;
  let opened: Promise<Cache | null> | null = null;
  const cache = () => (opened ??= storage.open(CACHE_NAME).catch(() => null));

  async function cached(url: string, read: () => Promise<Uint8Array | undefined>): Promise<Uint8Array | undefined> {
    const c = await cache();
    const hit = await c?.match(url).catch(() => undefined);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const bytes = await read();
    if (bytes && c) await c.put(url, new Response(bytes as Uint8Array<ArrayBuffer>)).catch(() => undefined);
    return bytes;
  }

  return {
    get: (key, opts) => cached(cacheKey(base, key), () => store.get(key, opts)),
    getRange: store.getRange && ((key, range, opts) => cached(cacheKey(base, key, range), () => store.getRange!(key, range, opts))),
  };
}
