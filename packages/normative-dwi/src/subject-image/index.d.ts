import type { NiftiHeader } from "@neurodesk/webapp-components/file-io/nifti";
import type { Grid } from "../model.js";

export interface SubjectMap {
  name: string;
  header: NiftiHeader;
  headerBytes: ArrayBuffer;
  data: Float64Array;
}

export class GridMismatchError extends Error {}

export function loadSubjectMap(name: string, bytes: ArrayBuffer | Uint8Array): Promise<SubjectMap>;
/**
 * For each store voxel, the flat index of the same voxel in the map. Throws
 * GridMismatchError, saying what differs, unless the map samples exactly the
 * template grid (any axis order or direction).
 */
export function gridMapping(map: Pick<SubjectMap, "name" | "header">, grid: Grid): Int32Array;
/** The map voxel containing world position `mm` (flat index in file order, and its centre), or null outside the image. */
export function voxelAtMm(header: NiftiHeader): (mm: ArrayLike<number>) => { index: number; centre: number[] } | null;
/** The map's values in store voxel order. */
export function gatherToStore(map: SubjectMap, mapping: Int32Array): Float64Array;
/** The z-map on the map's own grid, 0 where not valid, as .nii.gz bytes. */
export function scatterToNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, description: string): Uint8Array;
/** Display-only copy for the viewer, uncompressed: z saturated to ±limit, unscored voxels at `unscored`. */
export function displayNifti(z: Float64Array, valid: Uint8Array, mapping: Int32Array, map: SubjectMap, limit: number, unscored?: number): Uint8Array;

export interface BackgroundImage {
  name: string;
  header: NiftiHeader;
}

/** Read a structural image's header for display under the z-map: only the header is read (and gunzipped), the voxels are left to the viewer. */
export function readBackground(name: string, file: Blob): Promise<BackgroundImage>;
/** Fraction (0–1) of the z-map's world bounding box that the background's box covers, both placed by their raw affines as the viewer does. */
export function fieldOfViewOverlap(background: NiftiHeader, zmap: NiftiHeader): number;
/** The value unscored voxels carry in the display copy. */
export const UNSCORED_DISPLAY: number;
