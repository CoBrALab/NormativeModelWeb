# Benchmark summary (2026-10-18)

Synthetic marker: 241,373 modelled voxels of 304,327 (31 archives); 512 PCNtoolkit fits per toolkit, perturbed copies elsewhere. Old path timed on 620 voxels (every archive) and scaled to the marker; unpacking timed on every archive; new path measured on the whole marker. One CPU core.

Machine: AMD EPYC 9655 96-Core Processor, Linux 5.14.0-687.54.1.el9_8.x86_64; v24.15.0; Python 3.11.4 (PCNtoolkit 1.3.0), Python 3.12.4 (PCNtoolkit 1.4.0).

Same models, same subject: the fork's `compute_zscores` and the app kernel agree to max |Δz| = 2.4e-11 on the 620 timed voxels.

## Speed

| Typical use | PCNtoolkit 1.3.0 | PCNtoolkit 1.4.0 fork | nmzarr + app kernel | Gain vs 1.3.0 | Gain vs 1.4.0 |
|---|---|---|---|---|---|
| **Score one subject**, whole marker, `predict` (what proc_nm n01 calls) | 3.0 h (scaled) | 2.0 h (scaled) | **595 ms** (measured) | **18,291×** | 11,852× |
| **Score one subject**, whole marker, `compute_zscores` only | 34 min (scaled) | 33 min (scaled) | **595 ms** (measured) | **3,461×** | 3,375× |
| **Score 100 subjects**, whole marker, `predict` (what proc_nm n01 calls) | 3.3 h (scaled) | 2.0 h (scaled) | **8.6 s** (measured) | **1,394×** | 832× |
| **Score 100 subjects**, whole marker, `compute_zscores` only | 36 min (scaled) | 35 min (scaled) | **8.6 s** (measured) | **250×** | 245× |
| **Read** one marker's models (unpack + `NormativeModel.load` vs store read) | 3 min | 92.5 s | 516 ms | 367× | 179× |
| **Per voxel**, one subject: read; build inputs; z only; `predict` | 0.77 ms; 4.0 ms; 3.8 ms; 40.4 ms | 0.39 ms; 4.3 ms; 3.9 ms; 24.6 ms | 2.1 µs; none; 0.33 µs; — | 25,822× (z, all steps) | 25,216× |
| **Per subject** in a 100-subject cohort, whole marker | 119.5 s | 71.3 s | 81 ms + read once | | |

### Inside one `predict` call (ms per voxel, one subject)

| Step | PCNtoolkit 1.3.0 | PCNtoolkit 1.4.0 fork | Needed for a z-score? |
|---|---|---|---|
| `compute_zscores` | 3.8 | 3.9 | yes |
| `compute_centiles` | 5.1 | 5.1 | no |
| `compute_yhat` | 22.2 | 5.7 | no |
| `compute_baseline_logp` | 4.5 | 4.7 | no |
| `compute_logp` | 4.7 | 4.8 | no |
| other (`predict` bookkeeping) | 0.0 | 0.4 | |
| `NormData.from_dataframe` (before the call) | 4.0 | 4.3 | no |

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

