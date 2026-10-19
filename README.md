# NormativeModelWeb

This repository develops **Normative DWI** ([`apps/normative-dwi`](apps/normative-dwi/README.md)), a browser-only [Neurodesk webapp](https://github.com/neurodesk/webapps) that scores DTI/NODDI maps against the voxelwise normative models trained in `proc_nm`. The app is laid out exactly as it will sit in `neurodesk/webapps`, and this repository adds what it needs to build and test against the real upstream code.

## Layout

| Path | What |
|---|---|
| `apps/normative-dwi/` | The app, with its docs (`docs/`: glossary, ADRs) and Python oracle tools (`tools/`) |
| `vendor/webapps/` | `neurodesk/webapps` as a submodule, pinned to a commit; supplies the shared components, build scripts and checks |
| `upstream/registry.patch` | Everything the app changes upstream outside `apps/normative-dwi`: registry entries, citations, test lists, desktop workflow, e2e test data, changeset |
| `tools/link-upstream.sh` | Development wiring: installs the submodule, applies the patch, symlinks the app into it, installs this workspace |
| `tools/export-upstream.sh` | Makes an upstream-ready tree (see Upstreaming) |
| `benchmark/` | Speed and storage of the app's path (nmzarr store + kernel) against the training archives scored by PCNtoolkit, on synthetic models ([results](benchmark/RESULTS.md)). Not exported upstream |

`scripts`, `site`, `test-utils` and `eslint.config.js` are symlinks into `vendor/webapps`, so the app's `../../…` paths resolve as they will upstream. The shared components package is a member of this workspace, so the app depends on it with `workspace:*`. `@playwright/test` is linked to the submodule's copy, so upstream's test helpers and the app's specs load a single Playwright.

## Setup

You need Node ≥ 22.12 with corepack. pnpm 11.7 is pinned in `package.json`.

```bash
git submodule update --init
bash tools/link-upstream.sh
pnpm dev        # http://localhost:5173/normative-dwi/
pnpm test       # the app's Node tests
```

See the [app README](apps/normative-dwi/README.md) for the model store, the code, and the tests and their data variables.

## Results (2026-10-18)

| Check | What it compares | Tolerance | Result |
|---|---|---|---|
| `test/kernel.test.js` | CPU kernel vs PCNtoolkit's own `compute_zscores` on the training-zip models, ages first clamped to the training range (proc_nm's convention; native PCNtoolkit would extrapolate). 12,600 cases: random, shard-edge, swapped-Sex-column and worst-conditioned voxels (cond(A) up to 1.1e8), clamped ages, \|ys\| up to 12 | \|Δz\| < 1e-6 | ≤ 1.5e-13 |
| `test/kernel.test.js` (committed, runs in CI) | CPU kernel vs PCNtoolkit on models it fitted to simulated data (`test/reference/synthetic-pcn.json`): 8 voxels, 4 with swapped Sex columns, 144 cases incl. clamped ages and \|ys\| up to 10 | \|Δz\| < 1e-6 | pass |
| `test/kernel.test.js` | Cubic B-spline basis vs scipy, extrapolation included | 1e-12 | pass |
| `test/model-store.test.js` | Parameters read from the real store reproduce the fixtures | \|Δz\| < 1e-6 | pass |
| `test/end-to-end.test.js` | Whole brain: NIfTI in → z-map out vs `tools/make_reference_zmap.py` | rel. < 1e-6, same zero pattern | pass |
| `e2e/gpu-parity.spec.js` | WebGPU vs PCNtoolkit and vs CPU: the committed simulated reference (always), and the local fixtures | 1e-3 | ≤ 7.8e-6 / ≤ 1.3e-5 |
| `e2e/synthetic.spec.js` | Whole UI on a generated store, CPU and GPU, downloads vs the float64 kernel | 1e-6 / 1e-3 | 0 / ≤ 2.7e-6 |
| `e2e/app.spec.js` | Whole UI on a real subject, CPU and GPU, downloads vs reference | 1e-6 / 1e-3 | 0 / ≤ 9.0e-5 |

The GPU tolerance is 1e-3 everywhere; its relative form is |Δz| / max(1, |z|). The measured GPU values come from a software GPU (Chromium's SwiftShader), which is also what CI uses. The CPU is the default compute. To check a real GPU, run the specs with `NEURODESK_HARDWARE_GPU=1` on a machine that has one; `synthetic.spec.js` then asserts a hardware adapter.

The oracle is the PCNtoolkit fork that saved the models (`gdevenyi/PCNToolkit@4219457`). The latest PCNtoolkit release on PyPI (1.3.0) cannot score them: it builds a 10-column design where the saved models have 9 weights (`ValueError: shapes (36,10) and (9,) not aligned`). So there is no cross-check against a released version yet.

## Data

The model store, the parity fixtures (`apps/normative-dwi/test/fixtures/`, gitignored) and the reference z-maps all derive from UK Biobank or from subjects. None of them is in this repository, and none may be committed. The tests that need them skip without them. The browser tests and CI use a generated store with random parameters instead, and CI checks the kernel against PCNtoolkit's own fits to simulated data (`apps/normative-dwi/test/reference/`).

## Publishing the model store

The store's redistribution terms are still open. Once publication is cleared:

1. Run `apps/normative-dwi/tools/prepare_store_release.py`. It writes `models/normative-dwi.manifest.json`.
2. Upload its output to the `neurodeskorg/webapps` Hugging Face dataset.
3. Rerun it with `--base-url` set to the pinned resolve URL.
4. Set `DEFAULT_STORE_URL` in `apps/normative-dwi/src/config.ts`, and `model_manifest` in the registry patch.

## Upstreaming

```bash
git -C vendor/webapps worktree add --detach /tmp/webapps-upstream origin/main
bash tools/export-upstream.sh /tmp/webapps-upstream
cd /tmp/webapps-upstream && pnpm test:contracts && pnpm --filter normative-dwi build
```

`export-upstream.sh` does three things:
- copies `apps/normative-dwi` as a real directory, including its docs, tools, README and CHANGELOG, but no local data;
- applies `upstream/registry.patch`, which carries the registry and app-information entries, the test-list changes (`REQUIRED_METHODS`, `WITHOUT_COMMAND_LINE`), the offline-executables row, the desktop workflow, the e2e test-data entries and the changeset;
- regenerates `pnpm-lock.yaml`. The patch does not carry the lockfile, because its entries depend on the upstream commit; the regenerated lockfile belongs in the upstream PR.

The upstream pull request is that tree's diff, lockfile included.

On upstream `bfadef04` (the current pin), the root contract tests then pass (678/682), except:
- two that need the app's example (`app-examples` and `agentic-catalog`);
- two that fail on unmodified `bfadef04` in this environment too (Playwright discovery for calmar and easy-mp2rage).

Still open before the upstream PR:
- a pinned example in `examples.json`, locked with `scripts/lock-example-assets.mjs`;
- the public store;
- then `pnpm audit:interfaces`, `pnpm test:mobile` and `pnpm test:interface-workflows`.

## Licence

MIT; see [LICENSE](LICENSE). The licence does not cover the model store or the parity fixtures, which derive from UK Biobank data and are not distributed with this repository.
