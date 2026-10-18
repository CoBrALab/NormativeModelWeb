# NormativeModelWeb

**Normative DWI** (`apps/normative-dwi`) is a browser-only [Neurodesk webapp](https://github.com/neurodesk/webapps). It scores one person's DTI/NODDI maps against the voxelwise normative models trained in `proc_nm` and writes z-score maps. Nothing leaves the browser except the download of the model store.

- **Inputs:** FA, MD, ICVF, ISOVF and/or OD maps on the UK Biobank 2 mm grid (97×115×97), harmonised to UK Biobank, plus the subject's age and sex.
- **Models:** the `ukb_blr_agesex` nmzarr store from `proc_nm/opt_inference` (PCNtoolkit 1.4 BLR, sinh-arcsinh warp, one model per voxel per marker).
- **Output:** one `<MARKER>_zscore.nii.gz` per input map, on that map's grid, with 0 where nothing was scored. The viewer shows the z-map alone, in CET-D1 at ±3.

Terms are defined in [GLOSSARY.md](GLOSSARY.md), and design decisions are recorded in [docs/adr/](docs/adr/).

## Layout

| Path | What |
|---|---|
| `apps/normative-dwi/` | The app, laid out exactly as it will sit in `neurodesk/webapps` |
| `apps/normative-dwi/src/model-store/` | Opens an nmzarr store (URL or local folder) and reads one marker's models |
| `apps/normative-dwi/src/predictor/` | Z-score kernel: CPU float64 (the reference) and WebGPU float32 |
| `apps/normative-dwi/src/subject-image/` | NIfTI in/out and voxel matching through the affines |
| `tools/` | Python oracle and fixtures, reference z-maps, store release, local store server |
| `vendor/webapps/` | `neurodesk/webapps` (submodule), which supplies the shared components, build scripts and checks |
| `upstream/registry.patch` | The app's registry entries, applied to the submodule by `link-upstream` |

`scripts`, `site` and `eslint.config.js` are symlinks into `vendor/webapps`, so the app's `../../scripts/...` paths resolve exactly as they will upstream.

## Setup

You need Node ≥ 22.12 with corepack. pnpm 11.7 is pinned in `package.json`.

```bash
git submodule update --init
bash tools/link-upstream.sh   # installs vendor/webapps, applies the registry patch, links the app, installs this workspace
pnpm dev                      # http://localhost:5173/normative-dwi/
pnpm build
```

There is no public model store yet (see "Publishing the model store"). In the meantime, open **Advanced settings** and either choose a local copy of the store folder or enter the URL of a store served with Range requests and CORS:

```bash
node tools/serve-store.mjs /scratch/parent41/proc_nm/opt_inference/stores/ukb_blr_agesex.zarr 8765
```

## Tests

```bash
cd apps/normative-dwi
node --test test/*.test.js                                  # unit + PCNtoolkit parity (committed fixtures)
NMZARR_STORE=/path/to/ukb_blr_agesex.zarr node --test test/*.test.js        # + the real store
NMZARR_STORE=... NORMATIVE_E2E_DIR=/path/to/ref node --test test/*.test.js  # + whole-brain reference
npx playwright test                                         # built app in Chromium (smoke; + app.spec with the env vars above)
npx playwright test -c playwright.gpu.config.js             # WebGPU parity (SwiftShader when there is no GPU)
```

| Check | What it compares | Tolerance |
|---|---|---|
| `test/kernel.test.js` | CPU kernel vs **PCNtoolkit's own `compute_zscores`**, 9,600 cases (5 markers, flipped `be_map` voxels, clamped ages) | \|Δz\| < 1e-6 |
| `test/model-store.test.js` | Parameters read from the real store reproduce the fixtures | \|Δz\| < 1e-6 |
| `test/end-to-end.test.js` | Whole brain: NIfTI in → z-map NIfTI out vs `tools/make_reference_zmap.py` | rel. < 1e-6, identical zero pattern |
| `e2e/gpu-parity.spec.js` | WebGPU vs PCNtoolkit and vs CPU | rel. < 1e-4 |
| `e2e/app.spec.js` | The whole UI, CPU and GPU, downloads vs reference | 1e-6 / 1e-4 |

## Tools (Python)

The reference environment is `/scratch/parent41/voxel_nm/.venv_pcnGabe`; `tools/requirements.txt` pins the same versions, including the PCNtoolkit fork commit.

- `tools/make_fixtures.py` writes `apps/normative-dwi/test/fixtures/predict-<MARKER>.json` from the store and the original model zips. The y values are synthetic, so the fixtures are safe to commit.
- `tools/make_reference_zmap.py` writes whole-brain reference z-maps for one subject. They contain subject data, so never commit them.
- `tools/prepare_store_release.py` copies the store and writes `models/normative-dwi.manifest.json`.

## Publishing the model store

The store is derived from UK Biobank data, and its redistribution terms are still open. Once publication is cleared:

1. Run `tools/prepare_store_release.py`.
2. Upload the output to the `neurodeskorg/webapps` Hugging Face dataset.
3. Rerun the script with `--base-url` set to the pinned resolve URL.
4. Set `DEFAULT_STORE_URL` in `apps/normative-dwi/src/config.ts`.

## Upstreaming

Copy `apps/normative-dwi` and `models/normative-dwi.manifest.json` into a `neurodesk/webapps` checkout. Apply `upstream/registry.patch`, change the components dependency back to `workspace:*`, add an example to `examples.json`, and run the upstream gates (`pnpm audit:interfaces`, `pnpm test:mobile`, `pnpm test:interface-workflows`).
