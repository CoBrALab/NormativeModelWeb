// Build small NIfTI-1 images in memory for tests.
import { makeNifti } from "../src/synthetic-store.js";

export { makeNifti };

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
