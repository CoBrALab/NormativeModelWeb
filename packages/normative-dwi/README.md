# @neurodesk/normative-dwi

The science of the Normative DWI web app: it reads the model store, computes the float64 z-scores and reads and writes the images. The web app imports it from here, and a command line runs it in Node. The terms are defined in the app's [glossary](../../apps/normative-dwi/docs/GLOSSARY.md), and the design decisions are in its [ADRs](../../apps/normative-dwi/docs/adr/).

## Code

| Path | Role |
|---|---|
| `src/model-store/` | Opens an nmzarr store (format 0.1) from a URL, a folder or any zarrita store, checks its model design and layout, and reads one marker's parameters |
| `src/predictor/` | The float64 z-score kernel (the reference, [ADR-0001](../../apps/normative-dwi/docs/adr/0001-cpu-f64-reference-gpu-f32.md)) and its CPU adapter. The web app adds its WebGPU adapter, with the same interface |
| `src/subject-image/` | Reads a map, matches it to the store grid through the affines, and writes the z-map NIfTI |
| `src/score.js` | Scores one map against one marker's models. The web app and the command line both call it |
| `src/node.js` | Node only: a store in a local folder, maps and z-maps as files, `self-check` |
| `src/synthetic-store.js` | A small store with random parameters and maps on its grid, for `self-check` and the tests |
| `bin/normative-dwi.js` | The command line |
| `validation/` | The Python oracle: parity fixtures, the committed reference and whole-brain reference z-maps |
| `scripts/prepare_store_release.py` | Writes the store's release manifest |

The modules are plain JavaScript. Their types are in the `.d.ts` file next to each one.

## Command line

`normative-dwi` scores maps the way the web app does on its CPU, and writes the same files:

```bash
normative-dwi --store ukb_blr_agesex.zarr --age 72 --sex female -o results \
  sub-01_FA.nii.gz sub-01_MD.nii.gz sub-01_ICVF.nii.gz sub-01_ISOVF.nii.gz sub-01_OD.nii.gz
```

- **Store:** `--store` takes a local nmzarr store folder. There is no default yet: the UK Biobank-derived store is not cleared for redistribution.
- **Maps:** the inputs are FA, MD, ICVF, ISOVF or OD maps, in any subset. They must be on the UK Biobank 2 mm grid and harmonised to UK Biobank. The marker comes from the file name (`FA`, `MD`, `ICVF`/`NDI`, `ISOVF`/`FWF`/`FISO` or `OD`/`ODI` as a separate word). For any other name, give it with `--marker FILE=MARKER`, which takes the same names and aliases.
- **Output:** `<MARKER>_zscore.nii.gz` in the output folder (`-o`, by default the current folder), on the map's own grid, 0 where unscored. An existing file is never overwritten: the write itself refuses one (`wx`), even one that appears during the run.
- **Checks:** every input is checked before any is scored: the covariates, each map's marker, the store's markers and each map's grid. The z-maps are held until every map is scored, so a map that fails later (for example, a marker fitted with another design) stops the run with nothing written. If a write fails, the z-maps this run already wrote are removed.
- **Exit status:** 0 on success, 1 when the inputs or the scoring fail, 2 on a usage error (a missing or repeated option, an age not written as decimal digits, no map).
- **Messages:** progress and warnings go to standard error. One example is an age outside a model's training range, which is scored at the nearest end. The paths of the written files go to standard output.
- **Speed:** it runs on one CPU core. A whole brain with all five markers took about 4 s in our measurement.

`normative-dwi self-check` writes a generated store and map to a temporary folder, scores them, and checks the z-map against the kernel.

There is no packaged release yet. `release.json` and the `normative-dwi-native.yml` workflow come with the public store, together with a `download-models` command that fetches it. Until then, run the command line from a checkout:

```bash
node packages/normative-dwi/bin/normative-dwi.js --help
```

## Tests

```bash
pnpm --filter @neurodesk/normative-dwi test
```

The tests need no data:

- They check the kernel against PCNtoolkit's own fits to simulated data, in `test/reference/synthetic-pcn.json`, made by `validation/make_synthetic_reference.py`.
- They check the store reader and the image I/O on a generated store.
- They run the command line end to end on a generated store, and compare each z-map with the kernel voxel by voxel.

Tests that need real data skip without it:

- `test/fixtures/` comes from `validation/make_fixtures.py`. It holds UK Biobank-derived parameters, so it is never committed.
- `NMZARR_STORE` points at the real store.
- `NORMATIVE_E2E_DIR` points at the output of `validation/make_reference_zmap.py`. With it, both the module chain and the command line are compared with the reference z-maps of a real subject.
- `NORMATIVE_TEMPLATE_NIFTI` points at a UK Biobank 2 mm template.
