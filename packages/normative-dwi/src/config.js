// Input interpretation and output naming shared by the web app and the command line.
import { MARKERS } from "./model.js";

const MARKER_TOKENS = {
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
export function guessMarker(fileName) {
  const tokens = fileName.replace(/\.nii(\.gz)?$/i, "").toUpperCase().split(/[^A-Z0-9]+/);
  const found = new Set(tokens.map((t) => MARKER_TOKENS[t]).filter(Boolean));
  return found.size === 1 ? [...found][0] : null;
}

export function isMarker(value) {
  return MARKERS.includes(value);
}

/** The marker a name gives (FA, MD, ICVF, ISOVF or OD, or an alias such as NDI or ODI), in any case, or null. */
export function parseMarker(name) {
  return MARKER_TOKENS[name.trim().toUpperCase()] ?? null;
}

/**
 * Covariates from form, automation or command-line values, or null unless age
 * is a number above 0 and up to 120 and sex is female or male. The age must be
 * written as plain decimal digits: "66,5", "72y", "0x42" or "1e2" is refused,
 * not read as 66, 72, 66 or 100.
 */
export function parseCovariates(age, sex) {
  const text = String(age).trim();
  if (!/^\d+(\.\d+)?$/.test(text)) return null;
  const years = Number(text);
  if (years <= 0 || years > 120) return null;
  return sex === "female" || sex === "male" ? { age: years, sex } : null;
}

/** Output file name for a marker's z-map. */
export const zmapFileName = (marker) => `${marker}_zscore.nii.gz`;

/** Overall training range of age over the scored voxels, for the clamping warning. */
export function trainingRange(covRanges, valid) {
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
export function clampingNotice(age, clamped, range) {
  if (!clamped || !range) return null;
  return `Age ${age} is outside the model's ${Math.round(range.min)}–${Math.round(range.max)} range; scored at the nearest age`;
}
