import type { Marker, MarkerParams } from "./model.js";

/** A model with random parameters on a small grid, shaped like the ukb_blr_agesex design (not fitted to anything). */
export interface SyntheticModel {
  shape: [number, number, number];
  mask: Uint8Array;
  ijk: Int32Array;
  nVox: number;
  /** Store voxel index → world mm, row-major 4×4. */
  affine: number[];
  markers: Marker[];
  params: Record<string, Omit<MarkerParams, "marker" | "nVox" | "nKnots" | "beLevelColumns">>;
}

export const SYNTHETIC_MARKERS: Marker[];
export const SYNTHETIC_FAMILY: string;
export const SHARD_VOXELS: number;
export const CHUNK_VOXELS: number;

/** A float32 NIfTI-1 image in memory. dims [nx, ny, nz]; affine 4 rows; data indexed x-fastest. */
export function makeNifti(dims: readonly number[], affine: readonly (readonly number[])[], data: ArrayLike<number>, options?: { voxOffset?: number }): Uint8Array;
/** Deterministic PRNG (mulberry32) and a normal sampler. */
export function random(seed: number): { uniform(): number; normal(): number };
export function syntheticModel(options?: { shape?: [number, number, number]; seed?: number; markers?: Marker[] }): SyntheticModel;
/** Write the model as an nmzarr store folder, sharded and compressed like the real store; returns the folder. */
export function writeSyntheticStore(dir: string, model: SyntheticModel): Promise<string>;
/** A map of one marker on the model's grid, written in one of three layouts, with its values in store voxel order. */
export function syntheticMap(model: SyntheticModel, marker: Marker, seed?: number, layout?: "ras" | "minc" | "las"): { bytes: Uint8Array; y: Float64Array };
/** Store voxel at a world position (mm) of the model's grid, or -1. */
export function storeVoxelAt(model: SyntheticModel, mm: readonly [number, number, number]): number;
/** The store's parameters for one marker as MarkerParams. */
export function syntheticParams(model: SyntheticModel, marker: Marker): MarkerParams;
