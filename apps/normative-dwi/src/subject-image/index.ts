// SubjectImage: read a subject's map, check it is on the template-space grid,
// gather its values in store voxel order, and write a z-map back on the map's
// own grid. Maps may store the grid in any axis order or direction (MINC-
// derived NIfTI is often z-fastest): voxels are matched through the affines.
import {
  createNiftiFromData,
  decodeNiftiBuffer,
  extractNiftiHeader,
  isValidNifti1,
  readNiftiImageData,
  type NiftiHeader,
} from "@neurodesk/webapp-components/file-io/nifti";
import { gzipSync } from "fflate";
import type { Grid } from "../model.ts";

/** World-space tolerance (mm) when matching voxel centres. */
const TOLERANCE_MM = 1e-3;
/** NIfTI spatial unit codes (xyzt_units & 7) in millimetres; 0 (unknown) is read as mm. */
const UNIT_MM: Record<number, number> = { 1: 1000, 2: 1, 3: 0.001 };
const NIFTI_INTENT_ZSCORE = 5;

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
  if (!isValidNifti1(decoded)) throw new GridMismatchError(`${name}: not a NIfTI-1 image`);
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

const fmt = (values: number[]) => values.map((v) => Number(v.toFixed(2))).join("×");
const columnNorms = (rows: ArrayLike<number>[]) => [0, 1, 2].map((c) => Math.hypot(rows[0][c], rows[1][c], rows[2][c]));

/** The map's affine in millimetres, whatever its xyzt_units. */
function affineMm(header: NiftiHeader): number[][] {
  const scale = UNIT_MM[header.xyztUnits & 7] ?? 1;
  return [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => header.affine[r][c] * scale));
}

/**
 * For each store voxel, the flat index of the same voxel in the map.
 * Throws GridMismatchError, saying what differs, unless the map samples
 * exactly the template grid (any axis order or direction).
 */
export function gridMapping(map: Pick<SubjectMap, "name" | "header">, grid: Grid): Int32Array {
  const { header } = map;
  const dims = [header.nx, header.ny, header.nz];
  const storeRows = [0, 1, 2].map((r) => [0, 1, 2, 3].map((c) => grid.affine[4 * r + c]));
  const storeVoxel = columnNorms(storeRows);
  const mm = affineMm(header);
  const fail = (reason: string) =>
    new GridMismatchError(`${map.name}: not on the ${grid.shape.join("×")} template grid at ${fmt(storeVoxel)} mm; ${reason}`);

  if ([...dims].sort((p, q) => p - q).join() !== [...grid.shape].sort((p, q) => p - q).join()) throw fail(`its size is ${dims.join("×")}`);
  const mapVoxel = columnNorms(mm);
  if ([...mapVoxel].sort().some((v, i) => Math.abs(v - [...storeVoxel].sort()[i]) > TOLERANCE_MM)) throw fail(`its voxels are ${fmt(mapVoxel)} mm`);

  // M = inv(A_map) · A_store, store voxel → map voxel. It must be a signed
  // permutation with integer offsets for the two grids to coincide.
  const toMap = invert4(mm);
  const a = grid.affine;
  const M = toMap.map((row) => [0, 1, 2, 3].map((c) => row[0] * a[c] + row[1] * a[4 + c] + row[2] * a[8 + c] + (c === 3 ? row[3] : 0)));
  const tol = TOLERANCE_MM / Math.max(...mapVoxel, 1e-6);
  const near = (v: number, target: number) => Math.abs(v - target) <= tol;
  for (let r = 0; r < 3; r++) {
    const ones = M[r].slice(0, 3).filter((v) => near(Math.abs(v), 1)).length;
    const zeros = M[r].slice(0, 3).filter((v) => near(v, 0)).length;
    if (ones !== 1 || zeros !== 2) throw fail("its axes are rotated or oblique relative to the template");
  }
  // Where the store's first voxel lands in the map, in map voxels.
  const offset = M.map((row) => row[3]);
  if (offset.some((v) => !near(v, Math.round(v)))) throw fail(`its origin is off the template's voxel centres by ${fmt(offset.map((v) => (v - Math.round(v)) * mapVoxel[0]))} mm`);
  const R = M.map((row) => row.map(Math.round));
  const out = new Int32Array(grid.nVox);
  for (let v = 0; v < grid.nVox; v++) {
    const i = grid.ijk[3 * v], j = grid.ijk[3 * v + 1], k = grid.ijk[3 * v + 2];
    const p = R.map((row) => row[0] * i + row[1] * j + row[2] * k + row[3]);
    if (p.some((value, axis) => value < 0 || value >= dims[axis])) throw fail(`its origin is shifted by whole voxels (${fmt(R.map((row) => row[3]))})`);
    out[v] = p[0] + dims[0] * (p[1] + dims[1] * p[2]);
  }
  return out;
}

/**
 * The map voxel containing world position `mm`: its flat index in the map's
 * own (file) order and its centre in mm, or null outside the image. Viewers
 * report positions in world space; their voxel indices may be reoriented
 * (NiiVue's are RAS), so the map is indexed from mm.
 */
export function voxelAtMm(header: NiftiHeader): (mm: ArrayLike<number>) => { index: number; centre: number[] } | null {
  const toMm = affineMm(header);
  const toVoxel = invert4(toMm);
  const dims = [header.nx, header.ny, header.nz];
  return (mm) => {
    const p = toVoxel.map((row) => Math.round(row[0] * mm[0] + row[1] * mm[1] + row[2] * mm[2] + row[3]));
    if (p.some((value, axis) => !(value >= 0 && value < dims[axis]))) return null;
    const centre = toMm.map((row) => row[0] * p[0] + row[1] * p[1] + row[2] * p[2] + row[3]);
    return { index: p[0] + dims[0] * (p[1] + dims[1] * p[2]), centre };
  };
}

/** The map's values in store voxel order. */
export function gatherToStore(map: SubjectMap, mapping: Int32Array): Float64Array {
  const y = new Float64Array(mapping.length);
  for (let v = 0; v < mapping.length; v++) y[v] = map.data[mapping[v]];
  return y;
}

/**
 * A fresh NIfTI-1 header for a z-map: the map's geometry, no extensions, intent
 * NIFTI_INTENT_ZSCORE, and `description` (≤ 79 characters) in descrip. Nothing
 * else of the input (its intent, description, auxiliary file) carries over.
 */
function zHeader(map: SubjectMap, description: string): ArrayBuffer {
  const header = new ArrayBuffer(352);
  new Uint8Array(header).set(new Uint8Array(map.headerBytes, 0, 348));
  const view = new DataView(header);
  const little = map.header.littleEndian;
  view.setFloat32(56, 0, little); // intent_p1..p3
  view.setFloat32(60, 0, little);
  view.setFloat32(64, 0, little);
  view.setInt16(68, NIFTI_INTENT_ZSCORE, little);
  view.setFloat32(108, 352, little); // vox_offset: no extensions
  const text = (offset: number, length: number, value: string) => {
    const bytes = new Uint8Array(header, offset, length);
    bytes.fill(0);
    bytes.set(new TextEncoder().encode(value).slice(0, length - 1));
  };
  text(148, 80, description); // descrip
  text(228, 24, ""); // aux_file
  text(328, 16, "z-score"); // intent_name
  new Uint8Array(header, 348, 4).fill(0); // extension flag
  return header;
}

function scatter(values: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, fill: number, description: string): ArrayBuffer {
  const { nx, ny, nz } = map.header;
  const volume = new Float32Array(nx * ny * nz).fill(fill);
  for (let v = 0; v < mapping.length; v++) if (valid[v]) volume[mapping[v]] = values[v];
  return createNiftiFromData(volume, zHeader(map, description), { dims: [nx, ny, nz], calMin: -3, calMax: 3, clampCalMax: false });
}

/** The z-map on the map's own grid, 0 where not valid, as .nii.gz bytes (docs/adr/0002). */
export function scatterToNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, description: string): Uint8Array {
  return gzipSync(new Uint8Array(scatter(z, valid, mapping, map, 0, description)), { level: 6 });
}

/**
 * Display-only copy for the viewer, uncompressed: z saturated to ±limit, and
 * unscored voxels far below −limit (UNSCORED_DISPLAY) so the viewer's
 * transparent colormap entry hides them while z ≤ −limit draws in its end colour.
 */
export function displayNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, limit: number): Uint8Array {
  const saturated = z.map((value) => Math.min(limit, Math.max(-limit, value)));
  return new Uint8Array(scatter(saturated, valid, mapping, map, UNSCORED_DISPLAY, "display copy"));
}

/** The value unscored voxels carry in the display copy. */
export const UNSCORED_DISPLAY = -1000;
