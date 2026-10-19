# ADR-0001: CPU float64 is the reference predictor; WebGPU float32 is checked against it

- Status: accepted
- Date: 2026-10-18 (default changed from GPU to CPU in review)

The predictor has two adapters. The CPU adapter computes in float64 and must reproduce PCNtoolkit's z-scores to |Δz| < 1e-6: in CI on PCNtoolkit's own fits to simulated data (`packages/normative-dwi/test/reference/synthetic-pcn.json`, from `validation/make_synthetic_reference.py`), and locally on the parity fixtures (`packages/normative-dwi/test/fixtures`, made locally by `validation/make_fixtures.py` from the original model zips; not committed, as they hold UK Biobank-derived parameters). Both adapters clamp age to the voxel's training range before scoring, as proc_nm does; native PCNtoolkit would extrapolate the B-spline, so the fixtures check proc_nm's convention on top of PCNtoolkit. WebGPU has no float64, so the GPU adapter runs the same algorithm in float32 and is held to a looser tolerance against the CPU adapter, not against Python. The CPU is the default: it is exact, and a whole run takes about as long on either adapter, since reading the parameters dominates. Users can choose the GPU under Advanced settings; the CPU adapter is its fallback when the GPU fails or its device is lost mid-run. The app does not declare `ci.hardware_gpu` upstream: CI runs the GPU path on SwiftShader WebGPU (the same WGSL), and real-GPU parity is a manual check (`playwright.gpu.config.js` with `NEURODESK_HARDWARE_GPU=1`). The command line (`packages/normative-dwi`) has the CPU adapter only, so it always writes the reference z-scores.

## Considered options

- **GPU only**: rejected. A float32 9×9 solve plus the sinh-arcsinh warp loses digits, and nothing would tie the published numbers to PCNtoolkit.
- **Emulated double precision (df64) on the GPU**: rejected for v1. It is several times slower and harder to debug, for an accuracy that the CPU path already provides.

## Consequences

Changes to the algorithm must land in both adapters, and in the same order of operations, or the CPU↔GPU tolerance test fails. The measured GPU error and timing are recorded below as they are measured.

The Cholesky factor of each voxel's precision A is computed on the CPU in float64 and uploaded; the shader only does the forward substitution. The design is collinear (the B-spline basis, the intercept and the two sex columns each sum to one), so cond(A) reaches ~1e8 (ISOVF), and a float32 factorisation breaks down: 8 ISOVF voxels came out NaN in the first whole-brain run. With the float64 factor the variance term is accurate to float32 epsilon. The remaining GPU error comes from rounding the large, cancelling posterior mean weights to float32; it cannot be removed on the GPU side, hence the 1e-3 tolerance.

## Measurements (2026-10-18, Chromium headless, SwiftShader WebGPU)

| Check | CPU vs reference | GPU vs reference |
|---|---|---|
| Committed simulated reference, 144 cases vs PCNtoolkit (`test/kernel.test.js`, `e2e/gpu-parity.spec.js`; no local data needed) | ≤ 2.2e-14 | ≤ 7.8e-6 (relative) |
| Fixtures, 12,600 cases vs PCNtoolkit incl. cond(A) up to 1.1e8 and \|ys\| up to 12 (`e2e/gpu-parity.spec.js`) | ≤ 1.5e-13 | ≤ 1.3e-5 (relative) |
| Whole brain, real subject, 5 markers (`e2e/app.spec.js`) | bit-identical | ≤ 9.0e-5 (FA; others ≤ 5.3e-5) |
| Generated store, 3 markers (`e2e/synthetic.spec.js`, runs in CI) | bit-identical | ≤ 2.7e-6 |

The oracle is the fork that saved the models (`gdevenyi/PCNToolkit@4219457`); PCNtoolkit 1.3.0 from PyPI cannot load them (it builds a 10-column design for 9 saved weights).

Speed is not a reason to prefer the GPU here: the CPU kernel scores a full marker (304,327 voxels) in ~80 ms, and a complete 5-marker run in the app takes ~11 s on either adapter, almost all of it spent reading the 600 MB of parameters.
