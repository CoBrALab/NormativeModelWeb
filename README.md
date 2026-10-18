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
| `apps/normative-dwi/test/synthetic-store.js` | Generates a small store with random parameters (and maps on its grid) for tests that need no data |
| `tools/` | Python oracle and fixtures, reference z-maps, store release, upstream link/export scripts |
| `vendor/webapps/` | `neurodesk/webapps` (submodule, pinned), which supplies the shared components, build scripts and checks |
| `upstream/registry.patch` | Everything the app changes outside `apps/normative-dwi` upstream: registry entries, citations, test lists, the desktop workflow |

`scripts`, `site`, `test-utils` and `eslint.config.js` are symlinks into `vendor/webapps`, so the app's `../../…` paths resolve exactly as they will upstream. The shared components package is a member of this workspace, so the app depends on it with `workspace:*` as it will upstream; `@playwright/test` is linked to the submodule's copy so upstream's test helpers and the app's specs load one Playwright.

## Setup

You need Node ≥ 22.12 with corepack. pnpm 11.7 is pinned in `package.json`.

```bash
git submodule update --init
bash tools/link-upstream.sh   # installs vendor/webapps, applies the patch, links the app, installs this workspace
pnpm dev                      # http://localhost:5173/normative-dwi/
pnpm build
```

There is no public model store yet (see "Publishing the model store"). In the meantime, open **Advanced settings** and either choose a local copy of the store folder or enter the URL of a store served with Range requests and CORS:

```bash
node apps/normative-dwi/e2e/serve-store.mjs /scratch/parent41/proc_nm/opt_inference/stores/ukb_blr_agesex.zarr 8765
```

## Tests

```bash
cd apps/normative-dwi
node --test test/*.test.js           # needs no data: generated store, kernel, image I/O, predictors, config
npx playwright test                  # built app in Chromium: smoke + the whole UI on a generated store
```

Tests that need local data skip unless it is there:

| Variable / file | Enables | Made by |
|---|---|---|
| `test/fixtures/predict-*.json` (gitignored) | CPU (and GPU) parity with PCNtoolkit | `tools/make_fixtures.py` |
| `NMZARR_STORE=<…/ukb_blr_agesex.zarr>` | Reading the real store; its parameters reproduce the fixtures | `proc_nm/opt_inference` |
| `NORMATIVE_E2E_DIR=<dir>` (with `NMZARR_STORE`) | Whole-brain z-maps of a real subject, Node and UI (`e2e/app.spec.js`) | `tools/make_reference_zmap.py` |
| `NORMATIVE_TEMPLATE_NIFTI=<…/avg.020_2mm.nii.gz>` | A real UKB 2 mm template is matched to the store grid | proc_nm templates |

The GPU parity spec with the fixtures uses the dev server: `npx playwright test -c playwright.gpu.config.js` (SwiftShader when there is no GPU).

| Check | What it compares | Tolerance | Result (2026-10-18) |
|---|---|---|---|
| `test/kernel.test.js` | CPU kernel vs PCNtoolkit's own `compute_zscores` on the training-zip models, with ages first clamped to the training range (proc_nm's convention; native PCNtoolkit would extrapolate). 12,600 cases: random, shard-edge, swapped-Sex-column and worst-conditioned voxels (cond(A) up to 1.1e8), clamped ages, \|ys\| up to 12 | \|Δz\| < 1e-6 | ≤ 1.5e-13 |
| `test/kernel.test.js` | Cubic B-spline basis vs scipy, extrapolation included | 1e-12 | pass |
| `test/model-store.test.js` | Parameters read from the real store reproduce the fixtures | \|Δz\| < 1e-6 | pass |
| `test/end-to-end.test.js` | Whole brain: NIfTI in → z-map out vs `tools/make_reference_zmap.py` | rel. < 1e-6, same zero pattern | pass |
| `e2e/gpu-parity.spec.js` | WebGPU vs PCNtoolkit and vs CPU on the fixtures | rel. < 1e-3 | ≤ 1.3e-5 |
| `e2e/synthetic.spec.js` | Whole UI on a generated store, CPU and GPU, downloads vs the float64 kernel | 1e-6 / 1e-3 | 0 / ≤ 2.7e-6 |
| `e2e/app.spec.js` | Whole UI on a real subject, CPU and GPU, downloads vs reference | 1e-6 / 1e-3 | 0 / ≤ 9.0e-5 |

The oracle is the PCNtoolkit fork that saved the models (`gdevenyi/PCNToolkit@4219457`). The latest PCNtoolkit release on PyPI (1.3.0) cannot score them: it builds a 10-column design where the saved models have 9 weights (`ValueError: shapes (36,10) and (9,) not aligned`), so there is no released-version cross-check yet.

## Tools (Python)

The reference environment is `/scratch/parent41/voxel_nm/.venv_pcnGabe`; `tools/requirements.txt` pins the direct dependencies (including the PCNtoolkit fork commit) and `tools/requirements-lock.txt` the whole environment.

- `tools/make_fixtures.py` writes `apps/normative-dwi/test/fixtures/predict-<MARKER>.json` from the store and the original model zips (the command used is in its docstring). Only `y` is synthetic: the per-voxel parameters derive from UK Biobank, so the fixtures are gitignored and must never be committed.
- `tools/make_reference_zmap.py` writes whole-brain reference z-maps for one subject. They contain subject data, so never commit them.
- `tools/prepare_store_release.py` copies the store and writes `models/normative-dwi.manifest.json`.
- `tools/link-upstream.sh` wires the app into `vendor/webapps` for development; `tools/export-upstream.sh <webapps-checkout>` makes an upstream-ready tree (real app directory, patch applied, lockfile regenerated, no local data).

## Publishing the model store

The store is derived from UK Biobank data, and its redistribution terms are still open. Once publication is cleared:

1. Run `tools/prepare_store_release.py`.
2. Upload the output to the `neurodeskorg/webapps` Hugging Face dataset.
3. Rerun the script with `--base-url` set to the pinned resolve URL.
4. Set `DEFAULT_STORE_URL` in `apps/normative-dwi/src/config.ts`.

## Upstreaming

```bash
git -C vendor/webapps worktree add --detach /tmp/webapps-upstream origin/main
bash tools/export-upstream.sh /tmp/webapps-upstream
cd /tmp/webapps-upstream && pnpm test:contracts && pnpm --filter normative-dwi build
```

On upstream `ad4c57c` the root contract tests then pass except two that need the app's example (`app-examples`, `agentic-catalog`) and two that fail on unmodified `main` in this environment too (Playwright discovery for calmar and easy-mp2rage). Still open before an upstream PR: a pinned example in `examples.json` (and `scripts/lock-example-assets.mjs`), a public store (`DEFAULT_STORE_URL`, `model_manifest`), then `pnpm audit:interfaces`, `pnpm test:mobile` and `pnpm test:interface-workflows`.

## Licence

MIT, see [LICENSE](LICENSE). The model store and the parity fixtures are not covered: they derive from UK Biobank data and are not distributed with this repository.
