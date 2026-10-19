// ModelStore: open an nmzarr store (format 0.1) and read one marker's voxel
// models as typed arrays. Hides zarrita, sharding and codecs, consolidated
// metadata and where the bytes come from. Caching is the caller's: a URL
// source takes the fetch to use (the web app passes its persistent cache).
import * as zarr from "zarrita";
import { MARKERS } from "../model.js";
import { checkDesign } from "./design.js";
import { directoryStore } from "./directory-store.js";

export const FORMAT_VERSION = "0.1";

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
];

function baseStore(source) {
  switch (source.kind) {
    case "url": {
      const url = source.url.replace(/\/+$/, "");
      // Suffix ranges read a shard's index in one request (no HEAD first); the
      // store format requires hosts to support them.
      return { store: new zarr.FetchStore(url, { fetch: source.fetch, useSuffixRequest: true }), label: url };
    }
    case "directory":
      return { store: directoryStore(source.handle), label: source.handle.name };
    case "store":
      return { store: source.store, label: source.label };
  }
  throw new Error(`Unknown model store source ${JSON.stringify(source.kind)}`);
}

function bytesOf(data) {
  if (data instanceof Uint8Array) return data;
  if (data instanceof zarr.BoolArray) return new Uint8Array(data.buffer);
  throw new Error("Unexpected boolean array encoding");
}

export async function openModelStore(source) {
  const { store: raw, label } = baseStore(source);
  let store;
  try {
    store = await zarr.withConsolidatedMetadata(raw, { format: "v3" });
  } catch (error) {
    throw new Error(`${label} is not a readable model store (${error.message})`, { cause: error });
  }
  const root = await zarr.open.v3(zarr.root(store), { kind: "group" });
  const attrs = root.attrs;
  if (attrs.nmzarr_format_version !== FORMAT_VERSION) {
    throw new Error(`${label}: model store format ${String(attrs.nmzarr_format_version)} is not supported (expected ${FORMAT_VERSION})`);
  }
  const markers = attrs.markers.filter((m) => MARKERS.includes(m));
  const nVox = Number(attrs.n_vox);

  const [mask, affine, ijk] = await Promise.all(
    ["mask", "affine", "ijk"].map((name) => zarr.open.v3(root.resolve(name), { kind: "array" }).then((a) => zarr.get(a))),
  );
  const shape = mask.shape;
  if (shape.length !== 3) throw new Error(`${label}: mask must be 3-D`);
  const grid = {
    shape: [shape[0], shape[1], shape[2]],
    affine: Float64Array.from(affine.data),
    ijk: Int32Array.from(ijk.data),
    nVox,
  };
  if (grid.ijk.length !== 3 * nVox) throw new Error(`${label}: ijk does not hold ${nVox} voxels`);

  async function loadMarker(marker, options = {}) {
    if (!markers.includes(marker)) throw new Error(`${label} has no ${marker} models`);
    const group = await zarr.open.v3(root.resolve(marker), { kind: "group" });
    const groupAttrs = group.attrs;
    checkDesign(marker, groupAttrs);
    const columns = [...groupAttrs.be_level_columns];
    const nKnots = groupAttrs.basis_mean.nknots + 2 * groupAttrs.basis_mean.degree;
    const dims = { nKnots, nBe: columns.length, D: nKnots - 4 + 1 + columns.length };
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
    const out = {};
    for (const { field, array, width } of arrays) {
      const target = field.dtype === "float64" ? new Float64Array(nVox * width) : field.dtype === "int16" ? new Int16Array(nVox * width) : new Uint8Array(nVox);
      const perVoxelBytes = target.BYTES_PER_ELEMENT * width;
      for (let start = 0; start < nVox; start += step) {
        options.signal?.throwIfAborted();
        const stop = Math.min(start + step, nVox);
        const selection = [zarr.slice(start, stop), ...array.shape.slice(1).map(() => null)];
        const chunk = await zarr.get(array, selection, { signal: options.signal });
        const values = field.dtype === "bool" ? bytesOf(chunk.data) : chunk.data;
        target.set(values, start * width);
        doneBytes += (stop - start) * perVoxelBytes;
        options.onProgress?.(doneBytes / totalBytes);
      }
      out[field.key] = target;
    }

    // Each be_map entry of a modelled voxel picks a one-hot design column; one
    // out of range would silently overwrite another coefficient (or index past
    // it on the GPU). Unmodelled voxels hold the fill value (-1 in real stores).
    const { beMap, hasModel } = out;
    for (let v = 0; v < nVox; v++) {
      if (!hasModel[v]) continue;
      for (let c = 0; c < dims.nBe; c++) {
        const column = beMap[v * dims.nBe + c];
        if (column < 0 || column >= dims.nBe) throw new Error(`${marker}/be_map: voxel ${v} names design column ${column}, outside 0–${dims.nBe - 1}`);
      }
    }
    return { marker, nVox, nKnots, beLevelColumns: columns, ...out };
  }

  return { label, family: String(attrs.family), markers, grid, loadMarker };
}
