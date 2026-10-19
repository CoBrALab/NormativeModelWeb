// The one model design the predictors implement (kernel.ts), checked against a
// marker group's metadata before any parameter is read. A store fitted with a
// different PCNtoolkit configuration must be refused, not silently mis-scored.

type Attrs = Record<string, unknown>;

const REGRESSION: Record<string, unknown> = {
  type: "BLR",
  warp_name: "warpsinharcsinh",
  warp_reparam: true,
  heteroskedastic: true,
  fixed_effect: true,
  fixed_effect_slope: false,
  fixed_effect_var: false,
  ard: false,
  var_D: 2,
  n_hyp: 5,
};

const BASIS_MEAN: Record<string, unknown> = { basis_name: "bspline", degree: 3, include_linear: false };

// Options a writer may leave out: absent means PCNtoolkit's default, which is
// the value the kernel needs. Every other expected key must be present.
const ABSENT_MEANS: Record<string, unknown> = {
  fixed_effect_slope: false,
  fixed_effect_var: false,
  ard: false,
  include_linear: false,
  y_transform: null,
};

function mismatches(actual: Attrs | undefined, expected: Record<string, unknown>, prefix: string): string[] {
  const value = (key: string) => (actual && key in actual ? actual[key] : ABSENT_MEANS[key]);
  return Object.entries(expected)
    .filter(([key, wanted]) => value(key) !== wanted)
    .map(([key, wanted]) => `${prefix}${key} is ${actual && key in actual ? JSON.stringify(actual[key]) : "missing"}, expected ${JSON.stringify(wanted)}`);
}

/** Throws unless the marker group was fitted with the design kernel.ts implements. */
export function checkDesign(marker: string, attrs: Attrs): void {
  const problems = [
    ...mismatches(attrs.regression as Attrs, REGRESSION, "regression."),
    ...mismatches(attrs.basis_mean as Attrs, BASIS_MEAN, "basis_mean."),
    ...mismatches((attrs.basis_var as Attrs) ?? {}, { basis_name: "linear" }, "basis_var."),
    ...mismatches(attrs, { inscaler: "standardize", outscaler: "standardize", y_transform: null }, ""),
  ];
  const covariates = JSON.stringify(attrs.covariates);
  if (covariates !== '["Age"]') problems.push(`covariates are ${covariates}, expected ["Age"]`);
  const batchEffects = Object.keys((attrs.batch_effects as Attrs) ?? {});
  if (batchEffects.join() !== "Sex") problems.push(`batch effects are ${JSON.stringify(batchEffects)}, expected ["Sex"]`);
  if (problems.length) throw new Error(`${marker}: unsupported model design (${problems.join("; ")})`);
}
