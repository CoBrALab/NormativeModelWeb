# Benchmark: archives + PCNtoolkit against nmzarr + the app kernel

This benchmark scores the same voxelwise normative models in two ways and
compares speed and storage:

- **Old:** training archives (`Models_n<MARKER>_c<CHUNK>.zip`, one JSON model
  per voxel), scored through PCNtoolkit the way proc_nm's step 4 (`n01`) does
  it. The benchmark times PCNtoolkit **1.3.0**, the release proc_nm pins, and the
  **1.4.0 fork** (`gdevenyi/PCNToolkit@4219457`) that fitted the app's models.
- **New:** the same models converted to an **nmzarr** store (proc_nm
  `opt_inference`, Zarr v3), read by the app's store reader and scored by the
  app's float64 CPU kernel (`apps/normative-dwi/src/predictor/kernel.ts`).

The results and the explanation of where the gains come from are in
[RESULTS.md](RESULTS.md).

All data is synthetic. Nothing here comes from UK Biobank, so the results can
be committed and anyone can rerun them.

## What it does

`run.sh` runs these steps:

1. **`make_models.py`**, once per PCNtoolkit version, builds one marker at the
   size of the real FA store: 304,327 mask voxels, 241,373 modelled, in 31
   archives. The steps are:
   - Fit 512 voxel models to simulated data, using the design of the UK
     Biobank models. The design comes from the app's
     `tools/make_synthetic_reference.py`.
   - Fill every modelled voxel with a perturbed copy of a random fit. The
     perturbation keeps each copy a valid model (see `perturb()`).
   - Write each copy with PCNtoolkit's own JSON layout, byte for byte, so the
     archives have the size of real ones.

   Set `PROTOTYPES` to the number of modelled voxels to fit every voxel
   instead.
2. **nmzarr convert** turns the 1.4.0 archives into a store. PCNtoolkit 1.3.0
   builds a 10-column design that the converter and the app do not take, so
   only the fork's archives are converted.
3. **`measure_storage.py`** measures bytes and objects for the archives and the
   store, including the part the app downloads.
4. **`bench_new.mjs`** measures the whole marker: it reads it from the store
   and scores every voxel for 1 and for 100 subjects.
5. **`bench_old.py`**, once per PCNtoolkit version, follows `n01`:
   - Unpack every archive, and time all of them.
   - For 20 voxels per archive (620 in all), run `NormativeModel.load` →
     `NormData.from_dataframe` → `predict`, and separately `compute_zscores`,
     each for 1 and for 100 subjects.
   - Time each `compute_*` step inside `predict`.
   - Scale the per-voxel time to the marker's 241,373 modelled voxels.
6. **`report.py`** writes `results/summary.md` from the result JSONs.

Both paths score the same synthetic subjects (`subjects.py` and
`subjects.mjs`). The report checks that the fork's `compute_zscores` and the
app kernel give the same z-scores on the timed voxels.

Everything runs on one CPU core: BLAS thread counts are set to 1, and the
kernel is single-threaded.

## Requirements

- Node ≥ 22.12, with the app's dependencies installed (`bash tools/link-upstream.sh`).
- The nmzarr converter: proc_nm's `opt_inference/`. By default `run.sh` looks
  for it at `../proc_nm/opt_inference` next to this repository. Set
  `NMZARR_DIR` to point elsewhere.
- Two Python environments:

  | Variable | Contents | Requirements | Reference environment |
  |---|---|---|---|
  | `PCN13_PYTHON` | PCNtoolkit 1.3.0, Python 3.11 | `requirements-pcn13.txt` | `/scratch/parent41/voxel_nm/.venv_pcn13` |
  | `PCN14_PYTHON` | PCNtoolkit 1.4.0 fork, zarr, nibabel; Python 3.12 | `requirements-pcn14.txt` | `/scratch/parent41/voxel_nm/.venv_pcnGabe` |

  `bash benchmark/setup-envs.sh` creates both as `benchmark/.venv-pcn13` and
  `benchmark/.venv-pcn14`, which are `run.sh`'s defaults.
- About 6 GB of free disk for `benchmark/work/` (archives and store), and
  `/dev/shm` space for one unpacked archive (up to about 160 MB).

## Run

```bash
bash benchmark/setup-envs.sh      # once; or set PCN13_PYTHON / PCN14_PYTHON
bash benchmark/run.sh
```

About 30 minutes on one core: 8 minutes to build each toolkit's archives, a
few minutes for each old-path benchmark, and seconds for everything else.
`run.sh` skips the model building when `benchmark/work/` already holds the
models, so a rerun takes only the timing steps. Delete `benchmark/work/` to
rebuild.

Sizes can be changed through `N_VOX`, `PROTOTYPES`, `SAMPLE_PER_CHUNK` and
`SUBJECTS` (for example `SUBJECTS=1,10,1000`). For a quick check:

```bash
N_VOX=30000 PROTOTYPES=16 SAMPLE_PER_CHUNK=3 SUBJECTS=1,10 WORK=/tmp/nmb RESULTS=/tmp/nmb/results bash benchmark/run.sh
```

## Output

`benchmark/results/` holds:

| File | Contents |
|---|---|
| `old-pcn13.json`, `old-pcn14.json` | Per-voxel times by stage and by `compute_*` step, unpacking time of every archive, the extrapolated marker time, and the z of subject 0 at every timed voxel |
| `new.json` | Store open and read times, scoring times for each cohort size, and z at the same voxels |
| `storage.json` | Archive and store sizes, per array and per object, and the anatomy of one voxel's JSON |
| `manifest-*.json` | The synthetic marker: sizes, seed, PCNtoolkit version, and the timed voxels |
| `environment.json` | CPU, OS, Node, git commits of this repository and of nmzarr, and the load average |
| `summary.md` | The tables, generated by `report.py` |

## What is not measured

- The rest of `n01`'s per-voxel work (building the frames, statistics, warning
  capture) is not timed. Its real cost is higher than the old-path numbers
  here, so those numbers are a lower bound.
- The store is read from local disk with a warm page cache. In the browser, a
  first visit downloads the predict arrays (see *What the app downloads* in
  [RESULTS.md](RESULTS.md)), and on most connections that takes longer than
  the computation.
- NIfTI reading and writing and the WebGPU path are not timed.
