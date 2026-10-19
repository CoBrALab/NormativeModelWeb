import type { Covariates, Marker } from "./model.js";

/** The marker a file name names, or null when it names none or several. */
export function guessMarker(fileName: string): Marker | null;
export function isMarker(value: string): value is Marker;
/** Covariates from form, automation or command-line values, or null unless age > 0 and sex is female or male. */
export function parseCovariates(age: string, sex: string): Covariates | null;
/** Output file name for a marker's z-map. */
export function zmapFileName(marker: Marker): string;
/** Overall training range of age over the scored voxels, for the clamping warning. */
export function trainingRange(covRanges: Float64Array, valid: Uint8Array): { min: number; max: number } | null;
/** One status line about age clamping, or null when nothing was clamped. */
export function clampingNotice(age: number, clamped: number, range: { min: number; max: number } | null): string | null;
