# Results: nmzarr + the app kernel against archives + PCNtoolkit

Measured on 2026-10-19 with [`run.sh`](README.md) on one synthetic marker the
size of the real FA store: 241,373 modelled voxels out of 304,327 mask voxels,
in 31 archives. The machine was a shared AMD EPYC 9655 login node, using one
core. The raw numbers are in [`results/`](results/), and
[`results/summary.md`](results/summary.md) has every generated table.

- **Old:** proc_nm's step 4 (`n01`). It unpacks each training archive and,
  for every voxel, calls `NormativeModel.load`, then
  `NormData.from_dataframe`, then `predict`. Measured with PCNtoolkit
  **1.3.0** (the release proc_nm pins) and with the **1.4.0 fork** that
  fitted the app's models.
- **New:** the same models converted to an nmzarr store. The app's store
  reader loads them, and the app's float64 CPU predictor (`cpuPredictor().zmap`,
  which is what the browser calls) scores them.

**Headline.** On the same models, scoring one subject on one marker takes
**0.6–0.7 s** on the new path against **34 min** for the cheapest PCNtoolkit
call that gives z-scores (`compute_zscores` on the 1.4.0 fork): **2,800–3,300×
faster**. The pipeline as deployed (`predict` on 1.3.0, which also computes
centiles, log-likelihoods and `yhat`, on 1.3.0's own fits) takes 3.1 h, which
is 15,000–17,600×. The range runs from the first run with the store open (cold)
to the median of repeats (warm).

On the same models and the same two subjects (one female, one male), the fork
and the app kernel give the same z-scores. The largest difference is
|Δz| = 2.4e-11, over the 620 voxels timed on the old path. PCNtoolkit 1.3.0
fits its own models, which the kernel cannot score: its z differs from the
kernel's on the same voxels by up to 14.6, so the 1.3.0 rows compare different
fits.

## Speed

The new path's time is cold (the store open, the first read and the first
scoring run) or warm (the medians of 5 repeats, no open). The old path is one
pass, scaled from 620 voxels.

| Typical use | Old path | nmzarr + app kernel | Gain |
|---|---|---|---|
| **Score one subject**, one marker, **like for like**: 1.4.0 fork, same models, `compute_zscores` | 34 min | **0.74 s** cold, 0.63 s warm (0.59 s first read + 0.05 s open + 0.10 s scoring) | **2,800–3,300×** |
| Score one subject, one marker, as `n01` does it (`predict`), 1.4.0 fork | 2.0 h | 0.74 / 0.63 s | 9,700–11,300× |
| Score one subject, one marker, as `n01` does it (`predict`), 1.3.0 (its own D = 10 fits) | 3.1 h | 0.74 / 0.63 s | **15,000–17,600×** |
| Score one subject, one marker, 1.3.0 z-scores only | 35 min | 0.74 / 0.63 s | 2,800–3,300× |
| **Score 100 subjects**, one marker, like for like (fork `compute_zscores`) | 36 min | **10.7 s** cold, 10.6 s warm | **200×** |
| Score 100 subjects, one marker, `predict`: fork; 1.3.0 | 2.0 h; 3.4 h | 10.7 s | 680×; 1,100× |
| **Read** one marker's models | 1.3.0: 3.1 min (unpacking 23 s, measured on all 31 archives; `NormativeModel.load` 2.7 min, scaled). Fork: 97 s | **0.59 s** first read, 0.53 s warm | 310× vs 1.3.0; 160× vs the fork |
| Score one subject, **all five markers** | 1.3.0 `predict`: about 15 h | about 4 s | about 15,000× |
| **Per voxel**, one subject | read 0.77 ms; inputs (`NormData`) 4.0 ms; z 3.9 ms; or `predict` 41.3 ms (1.3.0) | read 2.2 µs; scoring 0.41 µs | z with read, inputs and scoring: 8.5 ms → 2.6 µs, **3,300×**; scoring alone: 4.0 ms (fork) → 0.41 µs, **9,800×** |

The old path's per-voxel time is stable: the fork's `predict` has a median of
29.5 ms and quartiles of 29.2–29.8 ms over the 620 timed voxels.

The app yields to the event loop after each block of 16,384 voxels, so that
scoring can be cancelled (`zmapCpuAsync`). That costs about 20 ms per subject
in Node and is included in these numbers: the 100-subject figure is 10.7 s with
the yields and 8.1 s without. A browser clamps nested timers to 4 ms, which
could add up to 76 ms per subject.

## Storage

| | Training archives | nmzarr store | Gain |
|---|---|---|---|
| **One marker on disk** | 1.90 GB (1.4.0 fork); 3.76 GB (1.3.0) | **147 MB** | **13×** vs the fork's archives; 26× vs 1.3.0's |
| Bytes per modelled voxel | 7,881 (fork); 15,586 (1.3.0) | 607 | 13× |
| Files | 31 zips holding 482,746 JSON files | 75 objects (15 arrays, 5 shards each) | |
| **What the app downloads** for one marker | whole archives, 1.90 GB | **128 MB** in 45 objects (the 9 arrays prediction reads) | **15×** |
| In memory once decoded | | 194 MB of typed arrays | |
| Real age+sex family, 5 markers (measured on the real models; see the cross-check below) | 8.8 GB, 155 archives | 640 MB | 14× |

## Where the speed comes from

For one subject, 1.3.0 spends 46.1 ms on each voxel, unpacking included:

| Part of the old path | ms per voxel | Share | In the new path |
|---|---|---|---|
| Unpack the archive (n01 does this into `/dev/shm`) | 0.09 | 0.2% | One store read for the whole marker: 2.2 µs per voxel, everything included |
| `NormativeModel.load`: parse 2 JSON files and build the model objects | 0.68 | 1.5% | (in the store read) |
| `NormData.from_dataframe`: build an xarray dataset for the voxel | 4.0 | 8.7% | None: the subject's map is already a `Float64Array` |
| `compute_zscores` | 3.9 | 8.5% | The kernel: 0.41 µs |
| `compute_centiles` | 5.2 | 11% | Not computed |
| `compute_yhat` | 22.7 | 49% | Not computed |
| `compute_baseline_logp` | 4.7 | 10% | Not computed |
| `compute_logp` | 4.8 | 10% | Not computed |

Change by change, the whole marker gets faster like this (1.3.0 on its own
fits; the warm time is used for the new path):

| Change | Per voxel | One marker | Gain of this step | Cumulative |
|---|---|---|---|---|
| Starting point: `n01`'s path, 1.3.0 `predict` | 46.1 ms | 3.1 h | | |
| 1. Compute only z | 8.6 ms | 35 min | 5.3× | 5.3× |
| 2. Read the Zarr store instead of zips of JSON | 7.9 ms | 32 min | 1.1× | 5.9× |
| 3. Score in the kernel instead of through `NormData` and PCNtoolkit | 2.6 µs (2.2 read + 0.41 scoring) | 0.63 s | 3,000× | **17,600×** |

### 1. PCNtoolkit computes much more than z (5×)

`predict` is the call `n01` makes. It fills five outputs: z-scores, centiles,
two log-likelihoods, and the predicted mean `yhat`. The app needs only z.
The other four outputs take 81% of the time. `compute_yhat` alone takes half,
because it integrates numerically over the warp: in 1.3.0 that means about
200 inversions of the sinh-arcsinh warp per voxel. The fork made `yhat` 4×
faster (5.8 ms), and that accounts for most of the gap between 1.3.0 and the
fork (46.0 against 29.7 ms per voxel). Even `compute_zscores` alone costs
8.5 ms per voxel, once reading and `NormData` are counted.

### 2. Per-voxel framework overhead (the big step)

On one voxel, the arithmetic of a z-score is about 350 floating-point
operations plus 5 transcendental functions:
- a 9 × 9 Cholesky factorisation;
- one forward substitution;
- 4 nonzero B-spline terms;
- `exp`, `sinh`, `asinh` and `sqrt`.

PCNtoolkit wraps that arithmetic in a general data model. Python function
calls per voxel, counted with cProfile on these models
([`count_calls.py`](count_calls.py), output in [`results/calls-*.json`](results/)):

| Call | PCNtoolkit 1.3.0 | 1.4.0 fork |
|---|---|---|
| `NormData.from_dataframe` | 20,000 | 19,700 |
| `compute_zscores` | 27,400 | 26,600 |
| `predict` | 278,700 | 183,300 |

Those calls do the following for each voxel:
- build an xarray `Dataset`, then scale, align, merge and copy it;
- map the batch effects;
- build the design matrix with scipy;
- call `scipy.linalg.solve` with its input checks.

All of this costs the same whatever the number of subjects: from 1 to 100
subjects, `compute_zscores` goes from 8.5 to 8.9 ms per voxel, so each extra
subject adds only 3.8 µs. The fixed cost per voxel is therefore about 2,300
times PCNtoolkit's own marginal cost per subject, and about 20,000 times the
kernel's whole time per voxel (0.41 µs).

The kernel ([`kernel.ts`](../apps/normative-dwi/src/predictor/kernel.ts)) is
one loop over the voxels that reads typed arrays:
- It evaluates the cubic B-spline only on the active knot span (4 basis
  functions, not 10).
- It factorises A from its stored upper triangle and solves in place, reusing
  a handful of buffers allocated once, so the loop allocates no memory.
- It does no work for unmodelled or missing voxels.

V8 compiles the loop to machine code. It takes 0.41 µs per voxel and subject
(0.33 µs without the event-loop yields of `zmapCpuAsync`, which add about 20 ms
per subject and marker).

### 3. The Zarr store: reading becomes cheap, and the kernel becomes possible

Reading one marker the old way means:
- unpacking 31 archives, which hold 482,746 JSON files;
- parsing 3.7 GB of JSON text (1.8 GB for the fork's archives);
- building a PCNtoolkit model object per voxel.

That takes 3.1 minutes. The store holds the same parameters as 9 arrays, with
the voxel as the first axis. Those 9 arrays are 45 objects and 128 MB. The
reader decompresses them (blosc/zstd) straight into typed arrays in 0.59 s
(0.53 s once warm), about 310× faster.

That alone is only 2% of the old path's time. The store's larger effect is
indirect: the kernel depends on it. With every voxel's `m`, `A`, hyperparameters, knots and
scalers in contiguous arrays, scoring a marker is one loop over memory, with
no per-voxel files, parsing or objects, and no PCNtoolkit. The same layout
lets the browser:
- fetch only the 9 arrays prediction needs, by HTTP range requests;
- check the model design once per marker;
- cache the store after the first visit.

After the refactor, reading the store (2.2 µs per voxel) costs more than
scoring (0.41 µs) when scoring one subject. Any further speedup for one
subject would have to come from reading, not from the kernel.

### Cohort size

The old path pays its per-voxel overhead once per voxel, whatever the number
of subjects. The new path reads the store once and then costs 0.10 s per
subject per marker. So the gain is largest for one subject (15,000–17,600×
against 1.3.0 `predict`) and smaller for 100 (1,100×).

The two cohort sizes also give a rough marginal cost for each extra subject.
That estimate comes from a small difference between two measurements, so treat
it as a trend:

| Path | Marginal cost per extra subject |
|---|---|
| 1.3.0 `predict` | about 10 s |
| 1.3.0 `compute_zscores` | about 0.9 s |
| App predictor | 0.10 s |

For very large cohorts, the gain therefore tends towards about 100× against
1.3.0 `predict`, and about 9× against z-scores only.

`n01` also runs one worker per core. The app, in a browser tab, uses one
core, and these numbers compare one core with one core.

## Where the storage gain comes from

Bytes per modelled voxel, from the fork's archive to the store:

| Stage | Bytes per voxel |
|---|---|
| Archive entry: two JSON files and their zip headers (`Stored`, no compression, as training writes them) | 7,881 |
| The JSON text alone | 7,550 |
| Only the numbers prediction uses (A, m, hyp, knots, response scaler), as compact JSON | 1,935 |
| The same 109 numbers as binary float64 (A as a full 9 × 9) | 872 |
| A as its upper triangle, plus the age scaler and range: the store's 79 values per voxel | 632 |
| Compressed with blosc (zstd, bit shuffle): the 9 arrays the app reads | 530 |
| The whole store, all 15 arrays, unmodelled rows included | 607 |

Where each step's saving comes from:
- **Settings repeated in every voxel (4×).** Three quarters of each voxel's
  JSON describes the model, not the fit. Each voxel carries:
  - the template regression model;
  - the optimiser settings;
  - the scratch directory it was saved from;
  - the batch-effect maps and covariate ranges;
  - every key name;
  - indentation.

  The store keeps settings that are the same in every voxel once, as group
  attributes. It also drops what prediction never reads: the training data's
  statistics and the optimiser's bookkeeping.
- **Text against binary (2.2×).** A float64 written in JSON takes about 18
  characters; in binary it takes 8 bytes.
- **Upper triangle of A (1.4× on the row).** A is symmetric, so the store
  keeps 45 of its 81 entries. The saved matrices are not exactly
  symmetric, but the difference is rounding: its effect on the predictive
  variance is at most 3.8e-11 relative (proc_nm `opt_inference`
  `docs/explanation/design.md`).
- **Compression (1.2× overall).** Fitted float64 values barely compress, about
  1.4×, because their low mantissa bits are noise. Structured arrays compress
  well: `be_map` 20×, `has_model` 20×, `N` 70×, and the knots 2.5× (the end
  knots repeat 4 times). The 21% of rows with no model hold fill values and
  cost almost nothing.

PCNtoolkit 1.3.0's archives are twice the size of the fork's, for three
reasons:
- Its design has one more column (D = 10).
- It saves two more D × D matrices per voxel, `Lambda_a` and `Sigma_a`. They
  are the training prior; prediction does not read them.
- It writes lists one number per line.

The store's sizes, per array:

| Array | Decoded | On disk | Read by the app |
|---|---|---|---|
| `A_triu` (45 per voxel) | 110 MB | 76 MB | yes |
| `m` (9) | 22 MB | 16 MB | yes |
| `mean_knots` (10) | 24 MB | 10 MB | yes |
| `hyp` (5) | 12 MB | 9 MB | yes |
| `inscaler`, `outscaler`, `cov_ranges`, `be_map`, `has_model` | 26 MB | 17 MB | yes |
| `be_cov_ranges`, `gamma`, `mean_minmax`, `var_minmax`, `nlZ`, `N` | 28 MB | 19 MB | no |

## Cross-check against the real models

These figures were measured on 2026-10-18 and 2026-10-19 on the UK Biobank models: the age+sex
family, its nmzarr store, and the `nm_priors` archives that proc_nm scores
with 1.3.0. They cannot be reproduced from this repository, because the
models are not in it.

| | Synthetic (this benchmark) | Real |
|---|---|---|
| Archive bytes per model, fork | 7,881 | 7,778 (FA, archive 0) |
| Archives per marker | 1.90 GB | about 1.8 GB (FA) |
| Store per marker | 147 MB | about 128 MB (640 MB for 5 markers) |
| Fork `predict` per voxel | 29.7 ms | 29.0 ms |
| 1.3.0 `predict` per voxel | 46.0 ms | 55 ms, on `nm_priors`, a larger design with more covariates and batch effects |
| App: first read of a marker | 0.59 s | 1.03 s (FA) |
| App: scoring one subject | 0.10 s (async predictor) | 0.08 s (synchronous `zmapCpu`, before the yields) |

Two of these match: the fork's `predict` per voxel, and the archive bytes per
voxel. The others differ. The synthetic store is about 15% larger than the real
one (so the 13× is conservative), because the perturbations put noise in every
value, including some that are near-constant in real models. The synthetic
first read is 43% faster than the real one. The 1.3.0 figures are on different
designs.

The support for using synthetic models at all is the design, not these
comparisons. The synthetic models have the real design (cubic B-spline with 4
knots, sinh-arcsinh warp, linear variance basis, Sex as the batch effect), so
they have the same D = 9, 10 knots and 5 hyperparameters per voxel, and the
app's `checkDesign` accepts them exactly as it accepts the real store. The
kernel's cost per voxel depends on those sizes, not on the parameter values.

## Method and limits

- **Synthetic models.**
  - **How they were made:** PCNtoolkit fitted 512 voxel models per version to
    simulated data, with the design of the UK Biobank models. The other
    voxels are perturbed copies of those fits. Each copy is still a valid
    model, written in PCNtoolkit's own JSON layout byte for byte.
  - **Why timing is unaffected:** timing does not depend on the parameter
    values, and PCNtoolkit's z-scores match the kernel's on the copies
    (|Δz| ≤ 2.4e-11).
  - **Reproducibility:** the L-BFGS-B fits differ between environments
    (numpy, scipy and BLAS builds), so a rerun elsewhere writes different
    parameters. It should give the same timings and sizes.
- **Extrapolation.** The old path's per-voxel steps were timed on 620 voxels,
  20 from each archive, and scaled to 241,373. Unpacking was timed on all 31
  archives. The new path was measured on the whole marker. The cold figure is
  the store open plus the first of 5 repeated runs; the warm figure uses the
  median of the 5 repeats. The old path is a single pass without repeats, but
  its per-voxel times vary little (see Speed).
- **Lower bound for `n01`.** The rest of `n01`'s per-voxel work (building the
  frames, statistics, warning capture) is not timed. The real pipeline is
  slower than the old-path numbers here.
- **Warm page cache.** The store was read from local disk, and the first read
  came after the store was written, so the operating system had cached it.
  In the browser, a first visit downloads 128 MB per marker (for example,
  about 10 s at 100 Mbit/s). On most connections that is longer than the
  computation. Later visits read from the browser cache.
- **Shared machine.** The machine was a shared login node, with a load
  average of about 12 on 192 cores. Each measurement used one core, and the
  BLAS thread counts were set to 1.
- **Unpacking in `/dev/shm`.** The old path unpacks each archive into
  `/dev/shm` (RAM), as `n01` does. On a network filesystem, unpacking 10,000
  small files took about 100 s per archive in a trial run, which would make the
  old path much slower. `SCRATCH` in `run.sh` chooses the directory.
- **Version 1.3.0 is not convertible.** PCNtoolkit 1.3.0 builds a 10-column
  design, which nmzarr and the app do not accept. Its numbers come from its
  own fits to the same simulated data. The fork's numbers come from exactly
  the models the store holds.
