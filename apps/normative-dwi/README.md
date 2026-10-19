# Normative DWI

Normative DWI scores one person's diffusion microstructure maps against voxelwise normative models of the UK Biobank population and writes a z-score map for each. Everything runs in the browser: the maps are never uploaded, and the only download is the model store.

- **Inputs:** FA and MD (DTI), and ICVF, ISOVF and OD (NODDI) maps, any subset. They must be on the UK Biobank 2 mm grid (97×115×97), resampled with a B-spline, ANLM-denoised and masked as in the models' training data, and harmonised to UK Biobank. Inputs also include the subject's age and sex.
- **Models:** a model store in nmzarr format 0.1 (Zarr v3), such as `ukb_blr_agesex`. Each voxel and marker has a PCNtoolkit 1.4.0 (fork gdevenyi/PCNToolkit@4219457, not released) Bayesian linear regression with a sinh-arcsinh warp, age-dependent noise, a cubic B-spline in age, and sex as a batch effect. The app refuses a store fitted with any other design.
- **Output:** `<MARKER>_zscore.nii.gz` on the input's grid, with NIfTI intent 5 (z-score). Voxels that were not scored are written as 0 ([ADR-0002](docs/adr/0002-missing-voxels-written-as-zero.md)).
- **Display:** colour saturates at |z| = L, 3 by default (as in the proc_nm figures). The **±|z|** box in the viewer toolbar sets L from 0.5 to 20. Downloads keep the true values.
- **Background (optional):** a T1-weighted or FLAIR image registered to UK Biobank space, at any resolution, shown under the z-map. Over it, each voxel's opacity is min(1, |z|/L), with L the colour limit, so near-normal voxels let the anatomy through; with **From |z|** off (viewer toolbar, top right), the **Overlay** slider sets one opacity for every scored voxel instead. Unscored voxels stay transparent either way. The display is interpolated between the 2 mm voxels; the readout gives each voxel's true z. It is for display only. The app checks that it covers the maps' field of view, not that it is registered.
- **Ages:** ages outside a voxel's training range (about 45–81) are scored at its nearest end, as the models' reference pipeline (CoBrALab proc_nm, not public yet) does, and the status line says so.
- **Compute:** the CPU path (float64) is the default and the reference, and reproduces PCNtoolkit to better than 1e-6. The GPU path (WebGPU, float32; Advanced settings) is tested to agree with it within 1e-3 ([ADR-0001](docs/adr/0001-cpu-f64-reference-gpu-f32.md)).
- **Command line:** [`@neurodesk/normative-dwi`](../../packages/normative-dwi/README.md) holds the store reader, the CPU kernel and the image I/O this app uses. Its `normative-dwi` command line writes the same z-maps from a local store folder.

Terms are defined in [docs/GLOSSARY.md](docs/GLOSSARY.md).

## Model store

Until a public store is pinned (`DEFAULT_STORE_URL` in `src/config.ts`), choose one under **Advanced settings**. You can pick a local copy of the store folder, or enter a URL served with HTTP Range requests and CORS:

```bash
node e2e/serve-store.mjs /path/to/ukb_blr_agesex.zarr 8765
```

Only stores at commit-pinned Hugging Face URLs (`https://huggingface.co/…/resolve/<40-hex commit>/…`) are cached in the browser, so a second visit downloads nothing. **Clear cached models** empties that cache.

## Code

The store reader, the CPU kernel, the image I/O and the per-map scoring step are in [`packages/normative-dwi`](../../packages/normative-dwi/README.md), with their tests and the Python oracle. This app adds:

| Path | Role |
|---|---|
| `src/main.ts` | The workflow: maps, covariates, store choice, runs, downloads and automation |
| `src/model-store/persistent-cache.ts` | Cache Storage for stores at commit-pinned Hugging Face URLs, passed to the store reader as its fetch |
| `src/predictor/` | The WebGPU float32 adapter, and the CPU fallback around it |
| `src/viewer.ts` | NiiVue view of one z-map: CET-D1 at ±L (L = 3 by default, the toolbar's ±\|z\| box), unscored voxels transparent, true-z readout; over a background, alpha modulation by a hidden layer: \|z\| for opacity from \|z\|, the scored mask for uniform opacity |

## Tests

```bash
pnpm test              # Node: GPU fallback, response cache, config (no data needed)
pnpm test:e2e          # Chromium on the built app: smoke + the whole UI on a generated store
```

The kernel, store and image I/O tests are in the [package](../../packages/normative-dwi/README.md#tests). The browser tests use its generated store, which has **random** parameters but is written like the real one (sharded, blosc), with maps on its grid in RAS, MINC and LAS layouts.

Tests that need real data skip without it:
- `NMZARR_STORE` points at the real store, and `NORMATIVE_E2E_DIR` at the output of `packages/normative-dwi/validation/make_reference_zmap.py`. With both, `e2e/app.spec.js` runs a whole subject through the UI, on the CPU and the GPU.
- The package's `test/fixtures/` (from `validation/make_fixtures.py`, never committed) adds the per-marker cases to the GPU parity check.

`playwright.gpu.config.js` runs the WebGPU path against PCNtoolkit: on the committed reference always, and on the fixtures when present.
