# Normative DWI

Normative DWI scores one person's diffusion microstructure maps against voxelwise normative models of the UK Biobank population and writes a z-score map for each. Everything runs in the browser: the maps are never uploaded, and the only download is the model store.

- **Inputs:** FA and MD (DTI), and ICVF, ISOVF and OD (NODDI) maps, any subset. They must be on the UK Biobank 2 mm grid (97×115×97), resampled with a B-spline, ANLM-denoised and masked as in the models' training data, and harmonised to UK Biobank. Inputs also include the subject's age and sex.
- **Models:** a model store in nmzarr format 0.1 (Zarr v3), such as `ukb_blr_agesex`. Each voxel and marker has a PCNtoolkit 1.4.0 (fork gdevenyi/PCNToolkit@4219457, not released) Bayesian linear regression with a sinh-arcsinh warp, age-dependent noise, a cubic B-spline in age, and sex as a batch effect. The app refuses a store fitted with any other design.
- **Output:** `<MARKER>_zscore.nii.gz` on the input's grid, with NIfTI intent 5 (z-score). Voxels that were not scored are written as 0 ([ADR-0002](docs/adr/0002-missing-voxels-written-as-zero.md)).
- **Ages:** ages outside a voxel's training range (about 45–81) are scored at its nearest end, as the models' reference pipeline (CoBrALab proc_nm, not public yet) does, and the status line says so.
- **Compute:** the CPU path (float64) is the reference and reproduces PCNtoolkit to better than 1e-6. The GPU path (WebGPU, float32) is tested to agree with it within 1e-3 ([ADR-0001](docs/adr/0001-cpu-f64-reference-gpu-f32.md)).

Terms are defined in [docs/GLOSSARY.md](docs/GLOSSARY.md).

## Model store

Until a public store is pinned (`DEFAULT_STORE_URL` in `src/config.ts`), choose one under **Advanced settings**. You can pick a local copy of the store folder, or enter a URL served with HTTP Range requests and CORS:

```bash
node e2e/serve-store.mjs /path/to/ukb_blr_agesex.zarr 8765
```

Only stores at commit-pinned Hugging Face URLs (`https://huggingface.co/…/resolve/<40-hex commit>/…`) are cached in the browser, so a second visit downloads nothing. **Clear cached models** empties that cache.

## Code

| Path | Role |
|---|---|
| `src/model-store/` | Opens a store (URL or local folder), checks its model design and layout, and reads one marker's parameters |
| `src/predictor/` | Z-score kernel: CPU float64 (reference) and WebGPU float32, with CPU fallback |
| `src/subject-image/` | Reads a map, matches it to the store grid through the affines, and writes the z-map NIfTI |
| `src/viewer.ts` | NiiVue view of one z-map: CET-D1 at ±3, unscored voxels transparent, true-z readout |
| `tools/` | Python oracle and helpers: parity fixtures, reference z-maps, store release |

## Tests

```bash
pnpm test              # Node: kernel, store, image I/O, predictors, config (no data needed)
pnpm test:e2e          # Chromium on the built app: smoke + the whole UI on a generated store
```

Two kinds of test data are committed or generated, and neither holds UK Biobank or subject data:
- `test/reference/synthetic-pcn.json` is from `tools/make_synthetic_reference.py`. PCNtoolkit fitted 8 voxel models to simulated data and scored 144 cases itself, so CI checks the kernel against an independent implementation.
- `test/synthetic-store.js` generates a small store with **random** parameters, written like the real one (sharded, blosc). It also generates maps on its grid, in RAS, MINC and LAS layouts.

Tests that need real data skip without it:
- `test/fixtures/` comes from `tools/make_fixtures.py`. It holds UK Biobank-derived parameters, so it is never committed.
- `NMZARR_STORE` points at the real store.
- `NORMATIVE_E2E_DIR` points at the output of `tools/make_reference_zmap.py`.
- `NORMATIVE_TEMPLATE_NIFTI` points at a UK Biobank 2 mm template.

`playwright.gpu.config.js` runs the WebGPU path against the fixtures.
