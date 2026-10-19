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

/** z-map display range (symmetric, saturating), as in the proc_nm figures; the default of the viewer's ±|z| box. */
export const Z_DISPLAY_LIMIT = 3;
/** Bounds of a user-set display limit. */
export const Z_LIMIT_RANGE = Object.freeze({ min: 0.5, max: 20 });

/** A display limit (|z| where the colour saturates) from the viewer's box, or null unless within Z_LIMIT_RANGE. */
export function parseZLimit(text: string): number | null {
  const value = Number(text.trim());
  return text.trim() !== "" && Number.isFinite(value) && value >= Z_LIMIT_RANGE.min && value <= Z_LIMIT_RANGE.max ? value : null;
}

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
