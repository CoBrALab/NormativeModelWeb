// A small nmzarr store (format 0.1) with RANDOM parameters, and maps on its
// grid, for whatever needs a whole store without data: the command line's
// self-check, browser e2e, GPU vs CPU parity, the desktop workflow. Nothing
// here comes from a fitted model; scientific parity with PCNtoolkit is tested
// separately (validation/make_fixtures.py).
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

/** A float32 NIfTI-1 image in memory. dims [nx, ny, nz]; affine 4 rows; data indexed x-fastest. */
export function makeNifti(dims, affine, data, { voxOffset = 352 } = {}) {
  const buffer = new ArrayBuffer(voxOffset + 4 * data.length);
  const view = new DataView(buffer);
  view.setInt32(0, 348, true);
  [3, dims[0], dims[1], dims[2], 1, 1, 1, 1].forEach((d, i) => view.setInt16(40 + 2 * i, d, true));
  view.setInt16(70, 16, true); // float32
  view.setInt16(72, 32, true);
  const spacing = [0, 1, 2].map((c) => Math.hypot(affine[0][c], affine[1][c], affine[2][c]));
  [1, ...spacing, 1, 1, 1, 1].forEach((p, i) => view.setFloat32(76 + 4 * i, p, true));
  view.setFloat32(108, voxOffset, true);
  view.setFloat32(112, 1, true);
  view.setInt16(254, 1, true); // sform
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) view.setFloat32(280 + 16 * r + 4 * c, affine[r][c], true);
  new Uint8Array(buffer, 344, 4).set([0x6e, 0x2b, 0x31, 0x00]); // "n+1\0"
  new Float32Array(buffer, voxOffset).set(data);
  return new Uint8Array(buffer);
}

export const SYNTHETIC_MARKERS = ["FA", "MD", "ICVF", "ISOVF", "OD"];
export const SYNTHETIC_FAMILY = "synthetic_blr_agesex";

/** Deterministic PRNG (mulberry32) and a normal sampler. */
export function random(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return { uniform, normal };
}

const DEGREE = 3;
const KNOTS = 10;
const D = KNOTS - DEGREE - 1 + 1 + 2;

function cubicBasis(knots, x) {
  // Cox–de Boor on the base interval only (training ages lie inside it).
  const n = knots.length - DEGREE - 1;
  let B = knots.slice(0, -1).map((t, i) => (t <= x && x < knots[i + 1] ? 1 : 0));
  for (let d = 1; d <= DEGREE; d++) {
    B = B.slice(0, -1).map((_, i) => {
      const left = knots[i + d] - knots[i] > 0 ? ((x - knots[i]) / (knots[i + d] - knots[i])) * B[i] : 0;
      const right = knots[i + d + 1] - knots[i + 1] > 0 ? ((knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1])) * B[i + 1] : 0;
      return left + right;
    });
  }
  return B.slice(0, n);
}

/**
 * Random voxel models on a small grid, shaped like the ukb_blr_agesex design.
 * The posterior precision A is built from a collinear training design, so it
 * is ill-conditioned the way real stores are.
 */
export function syntheticModel({ shape = [12, 14, 12], seed = 7, markers = SYNTHETIC_MARKERS } = {}) {
  const rng = random(seed);
  const [n0, n1, n2] = shape;
  const mask = new Uint8Array(n0 * n1 * n2);
  const ijk = [];
  const centre = shape.map((n) => (n - 1) / 2);
  for (let i = 0; i < n0; i++) for (let j = 0; j < n1; j++) for (let k = 0; k < n2; k++) {
    const r = Math.hypot((i - centre[0]) / centre[0], (j - centre[1]) / centre[1], (k - centre[2]) / centre[2]);
    if (r <= 0.95) {
      mask[(i * n1 + j) * n2 + k] = 1;
      ijk.push(i, j, k);
    }
  }
  const nVox = ijk.length / 3;
  // Store (MINC) axis order: array dim 0 is world z, like the real store.
  const affine = [0, 0, 2, -12, 0, 2, 0, -14, 2, 0, 0, -12, 0, 0, 0, 1];

  const params = {};
  for (const marker of markers) {
    const p = {
      hasModel: new Uint8Array(nVox), m: new Float64Array(nVox * D), aTriu: new Float64Array((nVox * D * (D + 1)) / 2),
      hyp: new Float64Array(nVox * 5), meanKnots: new Float64Array(nVox * KNOTS), inscaler: new Float64Array(nVox * 4),
      outscaler: new Float64Array(nVox * 4), beMap: new Int16Array(nVox * 2), covRanges: new Float64Array(nVox * 2),
    };
    for (let v = 0; v < nVox; v++) {
      p.hasModel[v] = rng.uniform() < 0.85 ? 1 : 0;
      const lo = 45 + rng.uniform(), hi = 80 + rng.uniform();
      const mean = 60 + 2 * rng.normal(), sd = 7 + rng.uniform();
      p.inscaler.set([mean, sd, lo, hi], 4 * v);
      p.covRanges.set([lo, hi], 2 * v);
      const xlo = (lo - mean) / sd, xhi = (hi - mean) / sd, pad = 0.2 * (xhi - xlo);
      const a = xlo - pad, b = xhi + pad;
      const knots = [a, a, a, ...Array.from({ length: 4 }, (_, i) => a + ((b - a) * i) / 3), b, b, b];
      p.meanKnots.set(knots, KNOTS * v);
      const flipped = rng.uniform() < 0.2;
      p.beMap.set(flipped ? [1, 0] : [0, 1], 2 * v);
      // Precision = β ΦᵀΦ + αI over a synthetic training sample (collinear design).
      const beta = 4, alpha = 1e-3;
      const A = Array.from({ length: D }, (_, r) => Array.from({ length: D }, (_, c) => (r === c ? alpha : 0)));
      for (let s = 0; s < 300; s++) {
        const x = xlo + (xhi - xlo) * rng.uniform();
        const phi = [...cubicBasis(knots, x), 1, 0, 0];
        phi[D - 2 + (rng.uniform() < 0.5 ? 0 : 1)] = 1;
        for (let r = 0; r < D; r++) for (let c = 0; c < D; c++) A[r][c] += beta * phi[r] * phi[c];
      }
      for (let r = 0, t = (v * D * (D + 1)) / 2; r < D; r++) for (let c = r; c < D; c++, t++) p.aTriu[t] = A[r][c];
      for (let d = 0; d < D; d++) p.m[v * D + d] = 0.3 * rng.normal();
      p.hyp.set([0.1 * rng.normal(), Math.log(4) + 0.1 * rng.normal(), 0.2 * rng.normal(), 0.1 * rng.normal(), Math.log(alpha)], 5 * v);
      const outMean = marker === "MD" ? 8e-4 : 0.45, outSd = marker === "MD" ? 1.5e-4 : 0.12;
      p.outscaler.set([outMean, outSd, outMean - 4 * outSd, outMean + 4 * outSd], 4 * v);
    }
    params[marker] = p;
  }
  return { shape, mask, ijk: Int32Array.from(ijk), nVox, affine, markers, params };
}

// Arrays are written the way the real store is (proc_nm/opt_inference
// nmzarr/convert.py): per-voxel arrays sharded along the voxel axis, inner
// chunks compressed with blosc (zstd, bitshuffle), a crc32c-checked shard index
// at the end; the root arrays as one zstd chunk. Shards are much smaller than
// the voxel count so readers cross shard boundaries and read partial shards.
export const SHARD_VOXELS = 512;
export const CHUNK_VOXELS = 128;

const TYPED = { bool: Uint8Array, uint8: Uint8Array, int16: Int16Array, int32: Int32Array, float64: Float64Array };

let crcTable;
function crc32c(bytes) {
  crcTable ??= Uint32Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0x82f63b78 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  let crc = 0xffffffff;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

const bytesCodec = (dataType) => (TYPED[dataType].BYTES_PER_ELEMENT === 1 ? { name: "bytes" } : { name: "bytes", configuration: { endian: "little" } });

function baseMeta(shape, dataType, chunkShape, codecs) {
  return {
    zarr_format: 3,
    node_type: "array",
    shape,
    data_type: dataType,
    chunk_grid: { name: "regular", configuration: { chunk_shape: chunkShape } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    fill_value: dataType === "bool" ? false : 0,
    codecs,
    attributes: {},
  };
}

async function codecs() {
  const [{ default: Blosc }, { default: Zstd }] = await Promise.all([import("numcodecs/blosc"), import("numcodecs/zstd")]);
  return { blosc: Blosc.fromConfig({ cname: "zstd", clevel: 5, shuffle: 2, blocksize: 0 }), zstd: Zstd.fromConfig({ level: 0 }) };
}

/** A per-voxel array as sharded files: [{ key, bytes }] plus its metadata. */
async function shardedArray(data, shape, dataType, blosc) {
  const width = shape.slice(1).reduce((a, b) => a * b, 1);
  const itemBytes = TYPED[dataType].BYTES_PER_ELEMENT;
  const raw = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const meta = baseMeta(shape, dataType, [SHARD_VOXELS, ...shape.slice(1)], [{
    name: "sharding_indexed",
    configuration: {
      chunk_shape: [CHUNK_VOXELS, ...shape.slice(1)],
      codecs: [bytesCodec(dataType), { name: "blosc", configuration: { typesize: itemBytes, cname: "zstd", clevel: 5, shuffle: "bitshuffle", blocksize: 0 } }],
      index_codecs: [{ name: "bytes", configuration: { endian: "little" } }, { name: "crc32c" }],
      index_location: "end",
    },
  }]);
  const perShard = SHARD_VOXELS / CHUNK_VOXELS;
  const rest = shape.slice(1).map(() => "0");
  const files = [];
  for (let shardStart = 0, s = 0; shardStart < shape[0]; shardStart += SHARD_VOXELS, s++) {
    const parts = [];
    const index = new BigUint64Array(2 * perShard).fill(0xffffffffffffffffn); // empty chunks
    let offset = 0;
    for (let c = 0; c < perShard; c++) {
      const start = shardStart + c * CHUNK_VOXELS;
      if (start >= shape[0]) break;
      // A chunk at the array's end is padded with the fill value to full size.
      const chunk = new Uint8Array(CHUNK_VOXELS * width * itemBytes);
      chunk.set(raw.subarray(start * width * itemBytes, Math.min(start + CHUNK_VOXELS, shape[0]) * width * itemBytes));
      const encoded = await blosc.encode(chunk);
      parts.push(encoded);
      index[2 * c] = BigInt(offset);
      index[2 * c + 1] = BigInt(encoded.length);
      offset += encoded.length;
    }
    const indexBytes = new Uint8Array(index.buffer);
    const checksum = new Uint8Array(4);
    new DataView(checksum.buffer).setUint32(0, crc32c(indexBytes), true);
    const shard = new Uint8Array(offset + indexBytes.length + 4);
    let at = 0;
    for (const part of parts) {
      shard.set(part, at);
      at += part.length;
    }
    shard.set(indexBytes, at);
    shard.set(checksum, at + indexBytes.length);
    files.push({ key: ["c", String(s), ...rest].join("/"), bytes: shard });
  }
  return { meta, files };
}

/** A root array as one zstd-compressed chunk, as in the real store. */
async function wholeArray(data, shape, dataType, zstd) {
  const meta = baseMeta(shape, dataType, shape, [bytesCodec(dataType), { name: "zstd", configuration: { level: 0, checksum: false } }]);
  const bytes = await zstd.encode(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return { meta, files: [{ key: ["c", ...shape.map(() => "0")].join("/"), bytes }] };
}

const MARKER_ATTRS = {
  covariates: ["Age"],
  batch_effects: { Sex: ["0", "1"] },
  be_level_columns: ["Sex=0", "Sex=1"],
  inscaler: "standardize",
  outscaler: "standardize",
  y_transform: null,
  regression: {
    type: "BLR", warp_name: "warpsinharcsinh", warp_reparam: true, heteroskedastic: true, fixed_effect: true,
    fixed_effect_slope: false, fixed_effect_var: false, ard: false, var_D: 2, n_hyp: 5, n_gamma: 2,
  },
  basis_mean: { basis_name: "bspline", degree: 3, nknots: 4, include_linear: false },
  basis_var: { basis_name: "linear" },
};

/** Write the model as an nmzarr store folder; returns the folder. */
export async function writeSyntheticStore(dir, model) {
  const { blosc, zstd } = await codecs();
  const nodes = {};
  const files = [];
  const add = (path, { meta, files: chunks }) => {
    nodes[path] = meta;
    files.push([join(path, "zarr.json"), JSON.stringify(meta)]);
    for (const { key, bytes } of chunks) files.push([join(path, key), bytes]);
  };
  add("mask", await wholeArray(model.mask, model.shape, "uint8", zstd));
  add("affine", await wholeArray(Float64Array.from(model.affine), [4, 4], "float64", zstd));
  add("ijk", await wholeArray(model.ijk, [model.nVox, 3], "int32", zstd));
  const n = model.nVox;
  for (const marker of model.markers) {
    const p = model.params[marker];
    const group = { zarr_format: 3, node_type: "group", attributes: { ...MARKER_ATTRS, dims: { D, n_hyp: 5, n_knots: KNOTS, n_cov: 1, n_be_levels: 2 } } };
    nodes[marker] = group;
    files.push([join(marker, "zarr.json"), JSON.stringify(group)]);
    for (const [name, data, shape, dataType] of [
      ["has_model", p.hasModel, [n], "bool"],
      ["m", p.m, [n, D], "float64"],
      ["A_triu", p.aTriu, [n, (D * (D + 1)) / 2], "float64"],
      ["hyp", p.hyp, [n, 5], "float64"],
      ["mean_knots", p.meanKnots, [n, KNOTS], "float64"],
      ["inscaler", p.inscaler, [n, 1, 4], "float64"],
      ["outscaler", p.outscaler, [n, 4], "float64"],
      ["be_map", p.beMap, [n, 2], "int16"],
      ["cov_ranges", p.covRanges, [n, 1, 2], "float64"],
    ]) add(`${marker}/${name}`, await shardedArray(data, shape, dataType, blosc));
  }
  const root = {
    zarr_format: 3,
    node_type: "group",
    attributes: { nmzarr_format_version: "0.1", family: SYNTHETIC_FAMILY, markers: model.markers, n_vox: n, synthetic: true },
    consolidated_metadata: { kind: "inline", must_understand: false, metadata: nodes },
  };
  files.push(["zarr.json", JSON.stringify(root)]);
  for (const [path, content] of files) {
    await mkdir(join(dir, path, ".."), { recursive: true });
    await writeFile(join(dir, path), content);
  }
  return dir;
}

/**
 * A map of one marker on the model's grid as a NIfTI. Values are random around
 * the voxel's response scaler; about 5% are 0 (missing). Returns { bytes, y }
 * with y in store voxel order. `layout` picks how the same grid is written:
 * "ras" x-fastest (the usual NIfTI), "minc" in the store's own order (z-fastest
 * in world terms) or "las" x-fastest with x running right to left.
 */
export function syntheticMap(model, marker, seed = 11, layout = "ras") {
  const rng = random(seed);
  const [n0, n1, n2] = model.shape; // store order: world z, y, x
  const y = new Float64Array(model.nVox);
  const p = model.params[marker];
  for (let v = 0; v < model.nVox; v++) {
    y[v] = rng.uniform() < 0.05 ? 0 : Math.fround(p.outscaler[4 * v] + p.outscaler[4 * v + 1] * 1.5 * rng.normal());
  }
  // Each layout: file dims, file index of store voxel (i, j, k), and file affine.
  const layouts = {
    ras: [[n2, n1, n0], (i, j, k) => k + n2 * (j + n1 * i), [[2, 0, 0, -12], [0, 2, 0, -14], [0, 0, 2, -12], [0, 0, 0, 1]]],
    minc: [[n0, n1, n2], (i, j, k) => i + n0 * (j + n1 * k), [[0, 0, 2, -12], [0, 2, 0, -14], [2, 0, 0, -12], [0, 0, 0, 1]]],
    las: [[n2, n1, n0], (i, j, k) => n2 - 1 - k + n2 * (j + n1 * i), [[-2, 0, 0, -12 + 2 * (n2 - 1)], [0, 2, 0, -14], [0, 0, 2, -12], [0, 0, 0, 1]]],
  };
  const [dims, fileIndex, affine] = layouts[layout];
  const data = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let v = 0; v < model.nVox; v++) {
    const [i, j, k] = model.ijk.subarray(3 * v, 3 * v + 3);
    data[fileIndex(i, j, k)] = y[v];
  }
  return { bytes: makeNifti(dims, affine, data), y };
}

/** Store voxel at a world position (mm) of the model's grid, or -1. */
export function storeVoxelAt(model, [x, y, z]) {
  const [i, j, k] = [(z + 12) / 2, (y + 14) / 2, (x + 12) / 2];
  for (let v = 0; v < model.nVox; v++) {
    if (model.ijk[3 * v] === i && model.ijk[3 * v + 1] === j && model.ijk[3 * v + 2] === k) return v;
  }
  return -1;
}

/** The store's parameters for one marker as MarkerParams (for comparing in Node). */
export function syntheticParams(model, marker) {
  return { marker, nVox: model.nVox, nKnots: KNOTS, beLevelColumns: ["Sex=0", "Sex=1"], ...model.params[marker] };
}
