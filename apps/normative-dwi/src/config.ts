// DOM-independent app configuration (unit-tested under Node). Input
// interpretation shared with the command line is in @neurodesk/normative-dwi.

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
