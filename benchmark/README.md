# Benchmark: archives + PCNtoolkit against nmzarr + the app kernel

This benchmark scores the same voxelwise normative models in two ways and
compares speed and storage:

- **Old:** training archives (`Models_n<MARKER>_c<CHUNK>.zip`, one JSON model
  per voxel), scored through PCNtoolkit the way proc_nm's step 4 (`n01`) does
  it. The benchmark times PCNtoolkit **1.3.0**, as proc_nm pins it (`pcntoolkit==1.3.0`; the reference environment has the git commit `amarquand/PCNtoolkit@3eac2b6`, 44 commits after the `v1.3.0` tag, which changed only files under `doc/`, so the package code is the 1.3.0 code), and the
  **1.4.0 fork** (`gdevenyi/PCNToolkit@4219457`) that fitted the app's models.
- **New:** the same models converted to an **nmzarr** store (proc_nm
  `opt_inference`, Zarr v3), read by the app's store reader and scored by the
  app's float64 CPU predictor (`apps/normative-dwi/src/predictor/`), the same
  call the browser makes.

The results and the explanation of where the gains come from are in
[RESULTS.md](RESULTS.md).

The models, the subjects and everything in `results/` are synthetic, so anyone
can rerun the benchmark. The only UK Biobank-derived numbers are the real FA
store's voxel counts (304,327 and 241,373, used as the default size), and the
aggregate file sizes and timings in the cross-check section of
[RESULTS.md](RESULTS.md). They describe model files, not model parameters or
participants. No script reads real models.

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
4. **`bench_new.mjs`** measures the whole marker: it opens the store, reads
   the marker, and scores every voxel for 1 and for 100 subjects through the
   app's `cpuPredictor().zmap`. It reports the first (cold) run with the store
   open, and the median of 5 repeats (warm).
5. **`bench_old.py`**, once per PCNtoolkit version, follows `n01`:
   - Unpack every archive, and time all of them.
   - For 20 voxels per archive (620 in all), run `NormativeModel.load` →
     `NormData.from_dataframe` → `predict`, and separately `compute_zscores`,
     each for 1 and for 100 subjects.
   - Time each `compute_*` step inside `predict`, and keep the time of every
     voxel (for the median and quartiles).
   - Scale the per-voxel time to the marker's 241,373 modelled voxels.

   `count_calls.py` counts the Python function calls PCNtoolkit makes per
   voxel (cProfile).
6. **`report.py`** writes `results/summary.md` from the result JSONs.

Both paths score the same synthetic subjects (`subjects.py` and
`subjects.mjs`): subject 0 is female (Sex level "0") and subject 1 is male.
The report checks that the fork's `compute_zscores` and the app kernel give
the same z-scores for both, on the timed voxels. For 1.3.0 it reports the
difference too; that is large, because 1.3.0 scores its own, different fits.

Everything runs on one CPU core: BLAS thread counts are set to 1, and the
kernel is single-threaded.

## Requirements

- Node ≥ 22.18 (it runs the app's TypeScript directly), with the app's dependencies installed (`bash tools/link-upstream.sh`).
- The nmzarr converter: proc_nm's `opt_inference/`. By default `run.sh` looks
  for it at `../proc_nm/opt_inference` next to this repository. Set
  `NMZARR_DIR` to point elsewhere. It is needed only to build the store.
- A system `python3` (standard library only), for `measure_storage.py`,
  `report.py` and the environment record. Set `NODE` if `node` is not the Node
  you want.
- Two Python environments:

  | Variable | Contents | Requirements |
  |---|---|---|
  | `PCN13_PYTHON` | PCNtoolkit 1.3.0, Python 3.11 | `requirements-pcn13.lock`: every package with hashes, except `pcntoolkit`, which is a git commit and has none, compiled from `requirements-pcn13.txt` with the versions in `reference-env-versions.txt` as constraints (`uv pip compile --python-version 3.11 --generate-hashes -c reference-env-versions.txt requirements-pcn13.txt -o requirements-pcn13.lock`) |
  | `PCN14_PYTHON` | PCNtoolkit 1.4.0 fork, zarr, nibabel; Python 3.12 | `requirements-pcn14.txt`, which includes the app's `tools/requirements-lock.txt` |

  `bash benchmark/setup-envs.sh` creates both as `benchmark/.venv-pcn13` and
  `benchmark/.venv-pcn14`, which are `run.sh`'s defaults.
- About 6 GB of free disk for `benchmark/work/` (archives and store). The old
  path also unpacks one archive at a time (up to about 160 MB) into `SCRATCH`,
  which defaults to `benchmark/work/scratch`. `n01` unpacks into `/dev/shm`;
  set `SCRATCH=/dev/shm` to do the same.

## Run

```bash
bash benchmark/setup-envs.sh      # once; or set PCN13_PYTHON / PCN14_PYTHON
SCRATCH=/dev/shm/nmbench bash benchmark/run.sh
```

The committed results were made with `SCRATCH=/dev/shm/...`, as `n01` unpacks
into `/dev/shm`. Unpacking 10,000 small files per archive on a network
filesystem is much slower, and the old path's time would include that.

About 30 minutes on one core: 8 minutes to build each toolkit's archives, a
few minutes for each old-path benchmark, and seconds for everything else.
`run.sh` reuses the models in `benchmark/work/` when their sizes have not
changed (`N_VOX`, `PROTOTYPES`, `SAMPLE_PER_CHUNK`), and rebuilds the store when
the models are newer than it, so a rerun takes only the timing steps. Delete
`benchmark/work/` to rebuild everything.

Sizes can be changed through `N_VOX`, `PROTOTYPES`, `SAMPLE_PER_CHUNK` and
`SUBJECTS` (for example `SUBJECTS=1,10,1000`). For a quick check:

```bash
N_VOX=30000 PROTOTYPES=16 SAMPLE_PER_CHUNK=3 SUBJECTS=1,10 WORK=benchmark/work-quick RESULTS=benchmark/work-quick/results bash benchmark/run.sh
```

## Output

`benchmark/results/` holds:

| File | Contents |
|---|---|
| `old-pcn13.json`, `old-pcn14.json` | Per-voxel times by stage and by `compute_*` step, unpacking time of each archive, the extrapolated marker time, and the z of a female and a male subject at every timed voxel |
| `new.json` | Store open time, read and scoring times (first run and repeats) for each cohort size, and z at the same voxels |
| `calls-pcn13.json`, `calls-pcn14.json` | Python function calls per voxel in PCNtoolkit |
| `storage.json` | Archive and store sizes, per array and per object, and the anatomy of one voxel's JSON |
| `manifest-*.json` | The synthetic marker: sizes, seed, PCNtoolkit version, and the timed voxels |
| `environment.json` | CPU, OS, Node, the load average, the git version of this repository and of nmzarr (`git describe --dirty`), and a hash of the benchmark's own files |
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
