// One map scored against its marker's voxel models: the steps the web app and
// the command line share, so their z-maps cannot drift apart.
import { clampingNotice, trainingRange } from "./config.js";
import { gatherToStore, gridMapping, scatterToNifti } from "./subject-image/index.js";

/** The descrip a z-map NIfTI carries (≤ 79 characters for the store families in use). */
export const zmapDescription = (family, marker, covariates) => `normative z (${family} ${marker}, age ${covariates.age}, ${covariates.sex})`;

export async function scoreMap({ store, map, marker, covariates, predictor, signal, onStatus, onLoadProgress }) {
  const mapping = gridMapping(map, store.grid);
  onStatus?.(`Loading ${marker} models…`);
  const params = await store.loadMarker(marker, { signal, onProgress: onLoadProgress });
  onStatus?.(`Scoring ${marker}…`);
  const { z, valid, clamped, kind } = await predictor.zmap(params, covariates, gatherToStore(map, mapping), signal);
  const scored = valid.reduce((n, x) => n + x, 0);
  const notice = clampingNotice(covariates.age, clamped, trainingRange(params.covRanges, valid));
  const nifti = scatterToNifti(z, valid, mapping, map, zmapDescription(store.family, marker, covariates));
  return { marker, kind, mapping, z, valid, scored, clamped, notice, nifti };
}
