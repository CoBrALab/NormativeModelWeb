import type { AsyncReadable } from "zarrita";
import type { Grid, Marker, MarkerParams } from "../model.js";
import type { DirectoryLike } from "./directory-store.js";

export const FORMAT_VERSION: "0.1";

export type StoreSource =
  /** Served with HTTP Range requests; `fetch` defaults to the global fetch. */
  | { kind: "url"; url: string; fetch?: (request: Request) => Promise<Response> }
  | { kind: "directory"; handle: DirectoryLike }
  /** Any zarrita store, under its own label (the command line's local folder, tests). */
  | { kind: "store"; store: AsyncReadable; label: string };

export interface ModelStore {
  label: string;
  family: string;
  markers: readonly Marker[];
  grid: Grid;
  /** Read every voxel model of one marker; onProgress gets a 0–1 fraction. */
  loadMarker(marker: Marker, options?: { onProgress?: (fraction: number) => void; signal?: AbortSignal }): Promise<MarkerParams>;
}

export function openModelStore(source: StoreSource): Promise<ModelStore>;
