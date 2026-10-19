// The one model design the predictors implement (kernel.js), checked against a
// marker group's metadata before any parameter is read. A store fitted with a
// different PCNtoolkit configuration must be refused, not silently mis-scored.

const REGRESSION = {
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

const BASIS_MEAN = { basis_name: "bspline", degree: 3, include_linear: false };

// Options a writer may leave out: absent means PCNtoolkit's default, which is
// the value the kernel needs. Every other expected key must be present.
const ABSENT_MEANS = {
  fixed_effect_slope: false,
  fixed_effect_var: false,
  ard: false,
  include_linear: false,
  y_transform: null,
};

function mismatches(actual, expected, prefix) {
  const value = (key) => (actual && key in actual ? actual[key] : ABSENT_MEANS[key]);
  return Object.entries(expected)
    .filter(([key, wanted]) => value(key) !== wanted)
    .map(([key, wanted]) => `${prefix}${key} is ${actual && key in actual ? JSON.stringify(actual[key]) : "missing"}, expected ${JSON.stringify(wanted)}`);
}

/** Throws unless the marker group was fitted with the design kernel.js implements. */
export function checkDesign(marker, attrs) {
  const problems = [
    ...mismatches(attrs.regression, REGRESSION, "regression."),
    ...mismatches(attrs.basis_mean, BASIS_MEAN, "basis_mean."),
    ...mismatches(attrs.basis_var ?? {}, { basis_name: "linear" }, "basis_var."),
    ...mismatches(attrs, { inscaler: "standardize", outscaler: "standardize", y_transform: null }, ""),
  ];
  const covariates = JSON.stringify(attrs.covariates);
  if (covariates !== '["Age"]') problems.push(`covariates are ${covariates}, expected ["Age"]`);
  const batchEffects = Object.keys(attrs.batch_effects ?? {});
  if (batchEffects.join() !== "Sex") problems.push(`batch effects are ${JSON.stringify(batchEffects)}, expected ["Sex"]`);
  if (problems.length) throw new Error(`${marker}: unsupported model design (${problems.join("; ")})`);
}
