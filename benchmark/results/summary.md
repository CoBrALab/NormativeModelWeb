# Benchmark summary (2026-10-19)

Synthetic marker: 241,373 modelled voxels of 304,327 (31 archives); 512 PCNtoolkit fits per toolkit, perturbed copies elsewhere. Old path timed on 620 voxels (every archive) and scaled to the marker; unpacking timed on every archive; new path measured on the whole marker. One CPU core.

Machine: AMD EPYC 9655 96-Core Processor, Linux 5.14.0-687.54.1.el9_8.x86_64; Node v24.15.0; Python 3.11.4 (PCNtoolkit 1.3.0), Python 3.12.4 (PCNtoolkit 1.4.0).

Same models, same subjects (one female, one male): the fork's `compute_zscores` and the app kernel agree to max |Δz| = 2.4e-11 on the 620 timed voxels. PCNtoolkit 1.3.0 scores its own D = 10 fits, which the kernel cannot score: its z differs from the kernel's on the same voxels by up to 14.6, so the 1.3.0 rows compare different fits.

## Speed

The new path's time is the first run, including the store open (cold); the warm time (median of repeats, no open) is in parentheses. Gains are given as cold–warm.

| Typical use | Old path | nmzarr + app kernel | Gain |
|---|---|---|---|
| **Score one subject**, whole marker, like for like: 1.4.0 fork, same models, `compute_zscores` | 34 min (scaled) | **743 ms** (633 ms warm) | **2,766×–3,250×** |
| Score one subject, whole marker, 1.3.0 `predict` (what proc_nm n01 calls) | 3.1 h (scaled) | **743 ms** (633 ms warm) | 14,960×–17,580× |
| Score one subject, whole marker, 1.4.0 fork `predict` (what proc_nm n01 calls) | 2.0 h (scaled) | **743 ms** (633 ms warm) | 9,655×–11,346× |
| Score one subject, whole marker, 1.3.0 `compute_zscores` only | 35 min (scaled) | **743 ms** (633 ms warm) | 2,797×–3,286× |
| **Score 100 subjects**, whole marker, like for like: 1.4.0 fork, same models, `compute_zscores` | 36 min (scaled) | **10.7 s** (10.6 s warm) | **200×** |
| Score 100 subjects, whole marker, 1.3.0 `predict` (what proc_nm n01 calls) | 3.4 h (scaled) | **10.7 s** (10.6 s warm) | 1,129× |
| Score 100 subjects, whole marker, 1.4.0 fork `predict` (what proc_nm n01 calls) | 2.0 h (scaled) | **10.7 s** (10.6 s warm) | 682× |
| Score 100 subjects, whole marker, 1.3.0 `compute_zscores` only | 36 min (scaled) | **10.7 s** (10.6 s warm) | 202× |
| **Read** one marker's models (unpack + `NormativeModel.load` vs store read) | 1.3.0 3 min; 1.4.0 fork 96.6 s | 590 ms first (533 ms warm) | 313× vs 1.3.0; 164× vs fork |
| **Per voxel**, one subject: read; build inputs; z only; `predict` | 1.3.0: 0.77 ms; 4.0 ms; 3.9 ms; 41.3 ms. Fork: 0.41 ms; 4.2 ms; 4.0 ms; 25.2 ms | read 2.2 µs; none; scoring 0.41 µs; — | z with read, inputs and scoring: 3,251× vs 1.3.0, 3,217× vs fork; scoring alone: 9,760× vs fork |
| **Per subject** in a 100-subject cohort, whole marker | 1.3.0 2 min; fork 73.2 s | 100 ms + read once | |

Spread of the old path: the fork's `predict` per voxel has median 29.5 ms, quartiles 29.2–29.8 ms over 620 voxels (one pass each).

### Inside one `predict` call (ms per voxel, one subject)

| Step | PCNtoolkit 1.3.0 | PCNtoolkit 1.4.0 fork | Needed for a z-score? |
|---|---|---|---|
| `compute_zscores` | 3.9 | 4.0 | yes |
| `compute_centiles` | 5.2 | 5.2 | no |
| `compute_yhat` | 22.7 | 5.8 | no |
| `compute_baseline_logp` | 4.7 | 4.8 | no |
| `compute_logp` | 4.8 | 5.0 | no |
| other (`predict` bookkeeping) | 0.0 | 0.4 | |
| `NormData.from_dataframe` (before the call) | 4.0 | 4.2 | no |

## Storage (one marker)

| | Archives, PCNtoolkit 1.3.0 | Archives, PCNtoolkit 1.4.0 fork | nmzarr store | Gain vs 1.4.0 archives |
|---|---|---|---|---|
| **On disk** | 3.76 GB | 1.90 GB | **147 MB** (+ 143.0 kB grid, shared by all markers) | **13×** |
| Bytes per modelled voxel | 15,586 | 7,881 | 607 | 13× |
| Files or objects | 31 zips holding 482,746 JSON files | 31 zips holding 482,746 JSON files | 75 objects (15 arrays) | |
| **What the app downloads** | whole archives | whole archives | **128 MB** in 45 objects (9 predict arrays) | 15× |
| In memory, decoded | | | 194 MB (typed arrays) | |
| One voxel's model | 15,246 B of JSON, D = 10 | 7,550 B of JSON, D = 9; 1,935 B of it are the numbers prediction uses | 79 float64 = 632 B before compression | 12× |

### Store arrays

| Array | Shape | dtype | Decoded | On disk | Ratio | Read by the app |
|---|---|---|---|---|---|---|
| `A_triu` | 304327 × 45 | float64 | 110 MB | 76 MB | 1.4 | yes |
| `m` | 304327 × 9 | float64 | 22 MB | 16 MB | 1.4 | yes |
| `mean_knots` | 304327 × 10 | float64 | 24 MB | 10 MB | 2.5 | yes |
| `hyp` | 304327 × 5 | float64 | 12 MB | 9 MB | 1.3 | yes |
| `outscaler` | 304327 × 4 | float64 | 10 MB | 7 MB | 1.4 | yes |
| `inscaler` | 304327 × 1 × 4 | float64 | 10 MB | 7 MB | 1.5 | yes |
| `be_cov_ranges` | 304327 × 2 × 1 × 2 | float64 | 10 MB | 6 MB | 1.6 | no |
| `gamma` | 304327 × 2 | float64 | 5 MB | 4 MB | 1.3 | no |
| `mean_minmax` | 304327 × 2 | float64 | 5 MB | 3 MB | 1.4 | no |
| `var_minmax` | 304327 × 2 | float64 | 5 MB | 3 MB | 1.4 | no |
| `cov_ranges` | 304327 × 1 × 2 | float64 | 5 MB | 3 MB | 1.6 | yes |
| `nlZ` | 304327 | float64 | 2 MB | 2 MB | 1.2 | no |
| `be_map` | 304327 × 2 | int16 | 1 MB | 61.9 kB | 19.7 | yes |
| `N` | 304327 | int32 | 1 MB | 17.1 kB | 71.1 | no |
| `has_model` | 304327 | bool | 304.3 kB | 15.4 kB | 19.7 | yes |

