import type { Covariates, Marker } from "./model.js";
import type { DirectoryLike } from "./model-store/directory-store.js";
import type { ModelStore } from "./model-store/index.js";

/** A DirectoryLike over a local folder. */
export function folderDirectory(path: string, name?: string): DirectoryLike;
/** Open the nmzarr model store in a local folder. */
export function openFolderStore(path: string): Promise<ModelStore>;
/** The marker of each map: from `markers` (keyed by the path as given, or its file name), else from the file name. */
export function mapMarkers(maps: readonly string[], markers?: Record<string, string>): { path: string; marker: Marker }[];

export interface ScoredFile {
  input: string;
  marker: Marker;
  output: string;
  scored: number;
  clamped: number;
}

/**
 * Score each map against the store and write OUTPUT/<MARKER>_zscore.nii.gz.
 * Every input is checked (marker, store, grid, existing outputs) before any is scored.
 */
export function scoreFiles(options: {
  store: string;
  maps: readonly string[];
  markers?: Record<string, string>;
  covariates: Covariates;
  output: string;
  onStatus?: (message: string) => void;
}): Promise<{ store: string; family: string; covariates: Covariates; files: ScoredFile[] }>;

/** Score a generated store and map in a temporary folder and check the written z-map against the float64 kernel. */
export function checkInstallation(): Promise<{ ok: true; node: string; voxels: number; scored: number }>;
