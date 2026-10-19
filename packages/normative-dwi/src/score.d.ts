import type { Covariates, Marker } from "./model.js";
import type { ModelStore } from "./model-store/index.js";
import type { Predictor, PredictorKind } from "./predictor/index.js";
import type { SubjectMap } from "./subject-image/index.js";

export interface ScoredMap {
  marker: Marker;
  /** The adapter that scored it. */
  kind: PredictorKind;
  /** Store voxel → flat index in the map. */
  mapping: Int32Array;
  /** Z per store voxel, 0 where not valid. */
  z: Float64Array;
  valid: Uint8Array;
  scored: number;
  /** Valid voxels whose age was clamped to the training range. */
  clamped: number;
  /** The clamping warning, or null. */
  notice: string | null;
  /** The z-map on the map's own grid, .nii.gz bytes. */
  nifti: Uint8Array;
}

export function zmapDescription(family: string, marker: Marker, covariates: Covariates): string;

/** Throws GridMismatchError when the map is not on the store's grid. */
export function scoreMap(options: {
  store: ModelStore;
  map: SubjectMap;
  marker: Marker;
  covariates: Covariates;
  predictor: Predictor;
  signal?: AbortSignal;
  onStatus?: (message: string) => void;
  /** 0–1 while the marker's models are read. */
  onLoadProgress?: (fraction: number) => void;
}): Promise<ScoredMap>;
