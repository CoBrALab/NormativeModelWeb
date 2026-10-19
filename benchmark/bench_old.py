"""Time the archive + PCNtoolkit path, the way proc_nm's n01 scores a marker.

For every archive: unpack it whole (n01 does this into /dev/shm), then for each
sampled voxel in it, NormativeModel.load -> NormData.from_dataframe -> predict
(n01's call: z-scores, centiles, log-likelihoods and yhat) or compute_zscores
(the cheapest PCNtoolkit call that gives z). Every archive is unpacked and
timed; the per-voxel steps are timed on the manifest's sample and scaled to
the marker's modelled voxels. n01's own per-voxel work (building the frames,
statistics, warning capture) is not included, so this is a lower bound for it.

Usage (in a PCNtoolkit environment):
  python -I benchmark/bench_old.py --models DIR --out results/old-<label>.json
"""

import argparse
import contextlib
import functools
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subjects  # noqa: E402

warnings.filterwarnings("ignore")
STEPS = ["compute_zscores", "compute_centiles", "compute_yhat", "compute_baseline_logp", "compute_logp"]


def instrument(cls, timings):
    """Time each top-level compute_* call that predict makes."""
    depth = [0]
    for name in STEPS:
        original = getattr(cls, name)

        @functools.wraps(original)
        def timed(self, *a, _orig=original, _name=name, **kw):
            depth[0] += 1
            t = time.perf_counter()
            try:
                return _orig(self, *a, **kw)
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    timings[_name] = timings.get(_name, 0.0) + time.perf_counter() - t

        setattr(cls, name, timed)


def score_voxel(NormativeModel, NormData, model_dir, mode, n_subj, v, scratch):
    """One voxel, one call. Returns (seconds per stage, z of every subject)."""
    t0 = time.perf_counter()
    model = NormativeModel.load(model_dir)
    model.saveresults = model.saveplots = model.savemodel = False
    model.evaluate_model = False
    model.save_dir = tempfile.mkdtemp(dir=scratch)
    t1 = time.perf_counter()
    out = model.outscalers["micro"]
    frame = pd.DataFrame({
        "uid": [f"s{s}" for s in range(n_subj)],
        "Age": [subjects.age(s) for s in range(n_subj)],
        "Sex": [subjects.sex_level(s) for s in range(n_subj)],
        "micro": [subjects.response(v, s, out.m, out.s) for s in range(n_subj)],
    })
    data = NormData.from_dataframe(name=f"v{v}", dataframe=frame, covariates=["Age"], batch_effects=["Sex"],
                                   response_vars=["micro"], subject_ids="uid")
    t2 = time.perf_counter()
    (model.predict if mode == "predict" else model.compute_zscores)(data)
    t3 = time.perf_counter()
    shutil.rmtree(model.save_dir, ignore_errors=True)
    z = data["Z"].sel(response_vars="micro").values.astype(np.float64)
    return {"load": t1 - t0, "normdata": t2 - t1, "call": t3 - t2}, z


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", type=Path, required=True, help="make_models.py output")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", default="1,100", help="cohort sizes to time, comma-separated")
    ap.add_argument("--scratch", default="/dev/shm" if os.path.isdir("/dev/shm") else None,
                    help="where archives are unpacked (n01 uses /dev/shm)")
    args = ap.parse_args()
    manifest = json.loads((args.models / "manifest.json").read_text())
    cohorts = [int(n) for n in args.subjects.split(",")]
    configs = [(mode, n) for n in cohorts for mode in ("predict", "zscores")]

    t = time.perf_counter()
    from pcntoolkit import NormativeModel, NormData
    import pcntoolkit.util.output
    import_s = time.perf_counter() - t
    pcntoolkit.util.output.Output.set_show_messages(False)
    step_times = {c: {} for c in configs}
    current = {}
    instrument(NormativeModel, current)

    scratch = tempfile.mkdtemp(prefix="nm-bench-old-", dir=args.scratch)
    sample = np.asarray(manifest["sample"])
    stage = {c: {"load": 0.0, "normdata": 0.0, "call": 0.0} for c in configs}
    z0 = {c: {} for c in configs}
    unzip_s, unzip_bytes, n_timed = 0.0, 0, 0
    try:
        for c in range(manifest["n_chunks"]):
            zip_path = args.models / f"Models_n{manifest['marker']}_c{c}.zip"
            extract = os.path.join(scratch, f"c{c}")
            t = time.perf_counter()
            shutil.unpack_archive(zip_path, extract, "zip")
            unzip_s += time.perf_counter() - t
            unzip_bytes += zip_path.stat().st_size
            lo, hi = c * manifest["training_chunk"], (c + 1) * manifest["training_chunk"]
            for v in sample[(sample >= lo) & (sample < hi)].tolist():
                for cfg in configs:
                    current.clear()
                    with contextlib.redirect_stdout(io.StringIO()):
                        times, z = score_voxel(NormativeModel, NormData, os.path.join(extract, f"model_{v}"),
                                               cfg[0], cfg[1], v, scratch)
                    for k, val in times.items():
                        stage[cfg][k] += val
                    for k, val in current.items():
                        step_times[cfg][k] = step_times[cfg].get(k, 0.0) + val
                    z0[cfg][str(v)] = float(z[0])
                n_timed += 1
            shutil.rmtree(extract, ignore_errors=True)
            print(f"  c{c}: {n_timed} voxels timed, unpacking {unzip_s:.1f}s so far", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    n_models = manifest["n_models"]
    results = {}
    for mode, n in configs:
        per_voxel = {k: 1e3 * val / n_timed for k, val in stage[(mode, n)].items()}
        per_voxel["total"] = sum(per_voxel.values())
        results[f"{mode}/{n}"] = {
            "mode": mode,
            "subjects": n,
            "ms_per_voxel": per_voxel,
            "ms_per_voxel_by_step": {k: 1e3 * val / n_timed for k, val in step_times[(mode, n)].items()},
            "marker_s": unzip_s + n_models * per_voxel["total"] / 1e3,
            "z_subject0": z0[(mode, n)],
        }
    doc = {
        "implementation": "archives + PCNtoolkit",
        "pcntoolkit": importlib.metadata.version("pcntoolkit"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "models": str(args.models),
        "n_models": n_models,
        "voxels_timed": n_timed,
        "import_s": import_s,
        "unzip_s_all_archives": unzip_s,
        "archive_bytes": unzip_bytes,
        "ms_per_voxel_read": 1e3 * unzip_s / n_models,
        "configs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    for key, r in results.items():
        print(f"{key}: {r['ms_per_voxel']['total']:.2f} ms/voxel, marker {r['marker_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
