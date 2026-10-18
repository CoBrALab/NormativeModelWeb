// ModelStore: open an nmzarr store (format 0.1) and read one marker's voxel
// models as typed arrays. Hides zarrita, sharding and codecs, consolidated
// metadata, persistent caching and where the bytes come from.
import * as zarr from "zarrita";
import { MARKERS, type Grid, type Marker, type MarkerParams } from "../model.ts";
import { checkDesign } from "./design.ts";
import { directoryStore, type DirectoryLike } from "./directory-store.ts";
import { cachingFetch } from "./persistent-cache.ts";

export const FORMAT_VERSION = "0.1";

type Readable = zarr.AsyncReadable;

export type StoreSource =
  | { kind: "url"; url: string }
  | { kind: "directory"; handle: DirectoryLike }
  /** Any zarrita store; used by tests. */
  | { kind: "store"; store: Readable; label: string };

export interface ModelStore {
  label: string;
  family: string;
  markers: readonly Marker[];
  grid: Grid;
  /** Read every voxel model of one marker; onProgress gets a 0–1 fraction. */
  loadMarker(marker: Marker, options?: { onProgress?: (fraction: number) => void; signal?: AbortSignal }): Promise<MarkerParams>;
}

/** Design sizes, from the group metadata: D = basis + intercept + batch columns. */
interface Dims {
  D: number;
  nKnots: number;
  nBe: number;
}

interface Field {
  key: keyof MarkerParams & string;
  array: string;
  dtype: string;
  width: (dims: Dims) => number;
}

// The predict arrays only (proc_nm/opt_inference docs/reference/format.md);
// N, nlZ, gamma, *_minmax and be_cov_ranges are never fetched. Every width is
// checked, so a store with another layout fails here instead of being read
// past each voxel's row.
const FIELDS = [
  { key: "hasModel", array: "has_model", dtype: "bool", width: () => 1 },
  { key: "m", array: "m", dtype: "float64", width: ({ D }) => D },
  { key: "aTriu", array: "A_triu", dtype: "float64", width: ({ D }) => (D * (D + 1)) / 2 },
  { key: "hyp", array: "hyp", dtype: "float64", width: () => 5 },
  { key: "meanKnots", array: "mean_knots", dtype: "float64", width: ({ nKnots }) => nKnots },
  { key: "inscaler", array: "inscaler", dtype: "float64", width: () => 4 },
  { key: "outscaler", array: "outscaler", dtype: "float64", width: () => 4 },
  { key: "beMap", array: "be_map", dtype: "int16", width: ({ nBe }) => nBe },
  { key: "covRanges", array: "cov_ranges", dtype: "float64", width: () => 2 },
] as const satisfies readonly Field[];

function baseStore(source: StoreSource): { store: Readable; label: string } {
  switch (source.kind) {
    case "url": {
      const url = source.url.replace(/\/+$/, "");
      return { store: new zarr.FetchStore(url, { fetch: cachingFetch() }), label: url };
    }
    case "directory":
      return { store: directoryStore(source.handle), label: source.handle.name };
    case "store":
      return { store: source.store, label: source.label };
  }
}

function bytesOf(data: unknown): Uint8Array {
  if (data instanceof Uint8Array) return data;
  if (data instanceof zarr.BoolArray) return new Uint8Array(data.buffer);
  throw new Error("Unexpected boolean array encoding");
}

export async function openModelStore(source: StoreSource): Promise<ModelStore> {
  const { store: raw, label } = baseStore(source);
  let store: Readable;
  try {
    store = await zarr.withConsolidatedMetadata(raw, { format: "v3" });
  } catch (error) {
    throw new Error(`${label} is not a readable model store (${(error as Error).message})`, { cause: error });
  }
  const root = await zarr.open.v3(zarr.root(store), { kind: "group" });
  const attrs = root.attrs as Record<string, unknown>;
  if (attrs.nmzarr_format_version !== FORMAT_VERSION) {
    throw new Error(`${label}: model store format ${String(attrs.nmzarr_format_version)} is not supported (expected ${FORMAT_VERSION})`);
  }
  const markers = (attrs.markers as string[]).filter((m): m is Marker => (MARKERS as readonly string[]).includes(m));
  const nVox = Number(attrs.n_vox);

  const [mask, affine, ijk] = await Promise.all(
    ["mask", "affine", "ijk"].map((name) => zarr.open.v3(root.resolve(name), { kind: "array" }).then((a) => zarr.get(a))),
  );
  const shape = mask.shape as number[];
  if (shape.length !== 3) throw new Error(`${label}: mask must be 3-D`);
  const grid: Grid = {
    shape: [shape[0], shape[1], shape[2]],
    affine: Float64Array.from(affine.data as Float64Array),
    ijk: Int32Array.from(ijk.data as Int32Array),
    nVox,
  };
  if (grid.ijk.length !== 3 * nVox) throw new Error(`${label}: ijk does not hold ${nVox} voxels`);

  async function loadMarker(marker: Marker, options: { onProgress?: (fraction: number) => void; signal?: AbortSignal } = {}): Promise<MarkerParams> {
    if (!markers.includes(marker)) throw new Error(`${label} has no ${marker} models`);
    const group = await zarr.open.v3(root.resolve(marker), { kind: "group" });
    const groupAttrs = group.attrs as { be_level_columns: string[]; basis_mean: { nknots: number; degree: number } } & Record<string, unknown>;
    checkDesign(marker, groupAttrs);
    const columns = [...groupAttrs.be_level_columns];
    const nKnots = groupAttrs.basis_mean.nknots + 2 * groupAttrs.basis_mean.degree;
    const dims: Dims = { nKnots, nBe: columns.length, D: nKnots - 4 + 1 + columns.length };
    const arrays = await Promise.all(
      FIELDS.map(async (field) => {
        const array = await zarr.open.v3(group.resolve(field.array), { kind: "array" });
        if (array.dtype !== field.dtype) throw new Error(`${marker}/${field.array}: expected ${field.dtype}, found ${array.dtype}`);
        if (array.shape[0] !== nVox) throw new Error(`${marker}/${field.array}: expected ${nVox} rows`);
        const width = array.shape.slice(1).reduce((a, b) => a * b, 1);
        const expected = field.width(dims);
        if (width !== expected) throw new Error(`${marker}/${field.array}: expected ${expected} values per voxel, found ${width}`);
        return { field, array, width };
      }),
    );

    // Read shard by shard so progress means something: A_triu alone is ~75 MB.
    const step = arrays[0].array.chunks[0];
    const totalBytes = arrays.reduce((sum, a) => sum + a.width * nVox * (a.field.dtype === "float64" ? 8 : a.field.dtype === "int16" ? 2 : 1), 0);
    let doneBytes = 0;
    const out: Partial<Record<string, Uint8Array | Float64Array | Int16Array>> = {};
    for (const { field, array, width } of arrays) {
      const target = field.dtype === "float64" ? new Float64Array(nVox * width) : field.dtype === "int16" ? new Int16Array(nVox * width) : new Uint8Array(nVox);
      const perVoxelBytes = target.BYTES_PER_ELEMENT * width;
      for (let start = 0; start < nVox; start += step) {
        options.signal?.throwIfAborted();
        const stop = Math.min(start + step, nVox);
        const selection = [zarr.slice(start, stop), ...array.shape.slice(1).map(() => null)];
        const chunk = await zarr.get(array, selection, { signal: options.signal });
        const values = field.dtype === "bool" ? bytesOf(chunk.data) : (chunk.data as Float64Array | Int16Array);
        target.set(values as never, start * width);
        doneBytes += (stop - start) * perVoxelBytes;
        options.onProgress?.(doneBytes / totalBytes);
      }
      out[field.key] = target;
    }

    return {
      marker,
      nVox,
      nKnots,
      beLevelColumns: columns,
      ...(out as Pick<MarkerParams, (typeof FIELDS)[number]["key"]>),
    };
  }

  return { label, family: String(attrs.family), markers, grid, loadMarker };
}
