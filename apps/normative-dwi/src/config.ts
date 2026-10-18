// DOM-independent app configuration and input interpretation (unit-tested under Node).
import { MARKERS, type Covariates, type Marker } from "./model.ts";

export const APP = Object.freeze({
  id: "normative-dwi",
  title: "Normative DWI",
});

/**
 * Public, pinned model store. null until the UK Biobank-derived parameters are
 * cleared for redistribution; users then provide a store URL or local folder.
 */
export const DEFAULT_STORE_URL: string | null = null;

/** z-map display range (symmetric, saturating), as in the proc_nm figures. */
export const Z_DISPLAY_LIMIT = 3;

const MARKER_TOKENS: Record<string, Marker> = {
  FA: "FA",
  MD: "MD",
  ICVF: "ICVF",
  NDI: "ICVF",
  ISOVF: "ISOVF",
  FWF: "ISOVF",
  FISO: "ISOVF",
  OD: "OD",
  ODI: "OD",
};

/** The marker a file name names, or null when it names none or several. */
export function guessMarker(fileName: string): Marker | null {
  const tokens = fileName.replace(/\.nii(\.gz)?$/i, "").toUpperCase().split(/[^A-Z0-9]+/);
  const found = new Set(tokens.map((t) => MARKER_TOKENS[t]).filter(Boolean));
  return found.size === 1 ? [...found][0] : null;
}

export function isMarker(value: string): value is Marker {
  return (MARKERS as readonly string[]).includes(value);
}

/** Covariates from form or automation values, or null unless age > 0 and sex is female or male. */
export function parseCovariates(age: string, sex: string): Covariates | null {
  const years = Number.parseFloat(age);
  if (!Number.isFinite(years) || years <= 0 || years > 120) return null;
  return sex === "female" || sex === "male" ? { age: years, sex } : null;
}

/** Output file name for a marker's z-map. */
export const zmapFileName = (marker: Marker) => `${marker}_zscore.nii.gz`;

/** Overall training range of age over the scored voxels, for the clamping warning. */
export function trainingRange(covRanges: Float64Array, valid: Uint8Array): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (let v = 0; v < valid.length; v++) {
    if (!valid[v]) continue;
    min = Math.min(min, covRanges[2 * v]);
    max = Math.max(max, covRanges[2 * v + 1]);
  }
  return Number.isFinite(min) ? { min, max } : null;
}

/** One status line about age clamping, or null when nothing was clamped. */
export function clampingNotice(age: number, clamped: number, range: { min: number; max: number } | null): string | null {
  if (!clamped || !range) return null;
  return `Age ${age} is outside the model's ${Math.round(range.min)}–${Math.round(range.max)} range; scored at the nearest age`;
}
