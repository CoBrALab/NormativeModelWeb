// SubjectImage: read a subject's map, check it is on the template-space grid,
// gather its values in store voxel order, and write a z-map back on the map's
// own grid. Maps may store the grid in any axis order or direction (MINC-
// derived NIfTI is often z-fastest): voxels are matched through the affines.
import {
  createNiftiFromData,
  decodeNiftiBuffer,
  extractNiftiHeader,
  readNiftiImageData,
  type NiftiHeader,
} from "@neurodesk/webapp-components/file-io/nifti";
import { gzipSync } from "fflate";
import type { Grid } from "../model.ts";

/** World-space tolerance (mm) when matching voxel centres. */
const TOLERANCE_MM = 1e-3;

export interface SubjectMap {
  name: string;
  header: NiftiHeader;
  headerBytes: ArrayBuffer;
  data: Float64Array;
}

export class GridMismatchError extends Error {
  override name = "GridMismatchError";
}

export async function loadSubjectMap(name: string, bytes: ArrayBuffer | Uint8Array): Promise<SubjectMap> {
  const decoded = await decodeNiftiBuffer(bytes);
  const image = readNiftiImageData(decoded, Float64Array);
  const frames = image.header.dims[0] > 3 ? image.header.dims.slice(4, 1 + image.header.dims[0]).reduce((a, b) => a * Math.max(1, b), 1) : 1;
  if (frames > 1) throw new GridMismatchError(`${name}: expected one 3-D map, found ${frames} volumes`);
  return { name, header: image.header, headerBytes: extractNiftiHeader(decoded), data: image.data };
}

function invert4(rows: ArrayLike<number>[]): number[][] {
  // Affine inverse: [R t; 0 1]⁻¹ = [R⁻¹ −R⁻¹t; 0 1]
  const r = [0, 1, 2].map((i) => [rows[i][0], rows[i][1], rows[i][2]]);
  const det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12) throw new GridMismatchError("the image affine is not invertible");
  const inv = [
    [r[1][1] * r[2][2] - r[1][2] * r[2][1], r[0][2] * r[2][1] - r[0][1] * r[2][2], r[0][1] * r[1][2] - r[0][2] * r[1][1]],
    [r[1][2] * r[2][0] - r[1][0] * r[2][2], r[0][0] * r[2][2] - r[0][2] * r[2][0], r[0][2] * r[1][0] - r[0][0] * r[1][2]],
    [r[1][0] * r[2][1] - r[1][1] * r[2][0], r[0][1] * r[2][0] - r[0][0] * r[2][1], r[0][0] * r[1][1] - r[0][1] * r[1][0]],
  ].map((row) => row.map((v) => v / det));
  const t = [rows[0][3], rows[1][3], rows[2][3]];
  return inv.map((row) => [...row, -(row[0] * t[0] + row[1] * t[1] + row[2] * t[2])]);
}

function describe(header: NiftiHeader): string {
  const [x, y, z] = header.voxelSize.map((v) => Number(v.toFixed(2)));
  return `${header.nx}×${header.ny}×${header.nz} at ${x}×${y}×${z} mm`;
}

/**
 * For each store voxel, the flat index of the same voxel in the map.
 * Throws GridMismatchError unless the map samples exactly the template grid.
 */
export function gridMapping(map: Pick<SubjectMap, "name" | "header">, grid: Grid): Int32Array {
  const { header } = map;
  const expected = `expected a ${grid.shape.join("×")} map at 2 mm in template space; found ${describe(header)}`;
  const toMap = invert4(header.affine);
  const a = grid.affine;
  // M = inv(A_map) · A_store, store voxel → map voxel. It must be a signed
  // permutation with integer offsets for the two grids to coincide.
  const M = toMap.map((row) => [0, 1, 2, 3].map((c) => row[0] * a[c] + row[1] * a[4 + c] + row[2] * a[8 + c] + (c === 3 ? row[3] : 0)));
  const voxelMm = Math.max(...header.voxelSize.map(Math.abs), 1e-6);
  const tol = TOLERANCE_MM / voxelMm;
  const near = (v: number, target: number) => Math.abs(v - target) <= tol;
  const dims = [header.nx, header.ny, header.nz];
  for (let r = 0; r < 3; r++) {
    const ones = M[r].slice(0, 3).filter((v) => near(Math.abs(v), 1)).length;
    const zeros = M[r].slice(0, 3).filter((v) => near(v, 0)).length;
    if (ones !== 1 || zeros !== 2 || !near(M[r][3], Math.round(M[r][3]))) throw new GridMismatchError(`${map.name}: ${expected}`);
  }
  const R = M.map((row) => row.map(Math.round));
  const out = new Int32Array(grid.nVox);
  for (let v = 0; v < grid.nVox; v++) {
    const i = grid.ijk[3 * v], j = grid.ijk[3 * v + 1], k = grid.ijk[3 * v + 2];
    const p = R.map((row) => row[0] * i + row[1] * j + row[2] * k + row[3]);
    if (p.some((value, axis) => value < 0 || value >= dims[axis])) throw new GridMismatchError(`${map.name}: ${expected}`);
    out[v] = p[0] + dims[0] * (p[1] + dims[1] * p[2]);
  }
  return out;
}

/** The map's values in store voxel order. */
export function gatherToStore(map: SubjectMap, mapping: Int32Array): Float64Array {
  const y = new Float64Array(mapping.length);
  for (let v = 0; v < mapping.length; v++) y[v] = map.data[mapping[v]];
  return y;
}

function scatter(values: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, fill: number): ArrayBuffer {
  const { nx, ny, nz } = map.header;
  const volume = new Float32Array(nx * ny * nz).fill(fill);
  for (let v = 0; v < mapping.length; v++) if (valid[v]) volume[mapping[v]] = values[v];
  return createNiftiFromData(volume, map.headerBytes, { dims: [nx, ny, nz], calMin: -3, calMax: 3, clampCalMax: false });
}

/** The z-map on the map's own grid and header, 0 where not valid, as .nii.gz bytes (docs/adr/0002). */
export function scatterToNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap): Uint8Array {
  return gzipSync(new Uint8Array(scatter(z, valid, mapping, map, 0)), { level: 6 });
}

/**
 * Display-only copy for the viewer, uncompressed: z saturated to ±limit, and
 * unscored voxels far below −limit so a viewer that hides values below its
 * minimum shows them transparent while z < −limit still draws in the end colour.
 */
export function displayNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, limit: number): Uint8Array {
  const saturated = z.map((value) => Math.min(limit, Math.max(-limit, value)));
  return new Uint8Array(scatter(saturated, valid, mapping, map, -1000 * limit));
}
