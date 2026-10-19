// Build small NIfTI-1 images in memory for tests.
/** dims [nx, ny, nz]; affine 4 rows; data indexed x-fastest. */
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

/** A small template space in store (MINC, z-fastest) order: shape [nz, ny, nx]. */
export function smallGrid() {
  const shape = [4, 5, 6]; // store dims: z, y, x
  const affine = Float64Array.from([0, 0, 2, -10, 0, 2, 0, -20, 2, 0, 0, -30, 0, 0, 0, 1]);
  const ijk = [];
  for (let i = 0; i < shape[0]; i++) for (let j = 0; j < shape[1]; j++) for (let k = 0; k < shape[2]; k++) if ((i + j + k) % 3) ijk.push(i, j, k);
  return { shape, affine, ijk: Int32Array.from(ijk), nVox: ijk.length / 3 };
}

/** World coordinate of store voxel v. */
export function worldOf(grid, v) {
  const [i, j, k] = grid.ijk.subarray(3 * v, 3 * v + 3);
  const a = grid.affine;
  return [0, 1, 2].map((r) => a[4 * r] * i + a[4 * r + 1] * j + a[4 * r + 2] * k + a[4 * r + 3]);
}

/** A float volume on `dims`/`affine` whose value encodes its own world position. */
export function worldCodedNifti(dims, affine) {
  const data = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let z = 0; z < dims[2]; z++) for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
    const w = [0, 1, 2].map((r) => affine[r][0] * x + affine[r][1] * y + affine[r][2] * z + affine[r][3]);
    data[x + dims[0] * (y + dims[1] * z)] = code(w);
  }
  return makeNifti(dims, affine, data);
}

export const code = ([x, y, z]) => 1 + (x + 100) + 1000 * (y + 100) + 1e6 * ((z + 100) / 10);
