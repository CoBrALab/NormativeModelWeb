// Domain types shared by the store reader, the predictors and the image I/O
// (declared in model.d.ts). Vocabulary: see apps/normative-dwi/docs/GLOSSARY.md.

export const MARKERS = Object.freeze(["FA", "MD", "ICVF", "ISOVF", "OD"]);

/** Store batch-effect level for a sex ("Sex=0" is female in the UK Biobank models). */
export function sexLevel(sex) {
  return sex === "female" ? "Sex=0" : "Sex=1";
}

/** Store design: the D = basis + intercept + one-hot batch columns. */
export function designSize(params) {
  return params.nKnots - 4 + 1 + params.beLevelColumns.length;
}
