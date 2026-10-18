"""Parity fixtures for the browser predictor, with PCNtoolkit as the oracle.

For a sample of voxels per marker this writes, to
test/fixtures/predict-<MARKER>.json (in this app):

  - the voxel's parameters exactly as the browser reads them from the nmzarr
    store (the predict arrays only), and
  - cases {age, sex, y} with the z-score PCNtoolkit itself computes from the
    ORIGINAL model in the training zip (NormativeModel.load + compute_zscores).

So the fixtures check the whole chain -- store layout, be_map, scalers,
B-spline, warp -- not a reimplementation of it. Ages outside the voxel's
covariate range are clamped before PCNtoolkit sees them, as proc_nm does
(scripts/normative/n01_transfer_models.py, clamp_to_ranges); the fixture keeps
the raw age so the browser must clamp too. So the cases check proc_nm's
convention on top of PCNtoolkit; native PCNtoolkit would extrapolate the
B-spline instead (that branch is tested against scipy in test/kernel.test.js).

The voxels sampled are random, plus every shard's first and last modelled
voxel, voxels with a swapped Sex column, and the worst-conditioned precision
matrices; the cases include large |ys| (deep in the warp's tails).

DATA: only y is synthetic. The per-voxel parameters (m, A_triu, hyp, scalers,
ranges) are copied from the store and derive from UK Biobank, which is not
cleared for redistribution. The output directory is gitignored: never commit it.

Usage (reference env: /scratch/parent41/voxel_nm/.venv_pcnGabe); the
fixtures used for the documented results were made with:
  python -I apps/normative-dwi/tools/make_fixtures.py \
    --store /scratch/parent41/proc_nm/opt_inference/stores/ukb_blr_agesex.zarr \
    --zips /scratch/parent41/voxel_nm/Analyses/norm_models_BLR_agesex/results/models_zip
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PREDICT_ARRAYS = ["has_model", "m", "A_triu", "hyp", "mean_knots", "inscaler",
                  "outscaler", "be_map", "cov_ranges"]
SEXES = ["0", "1"]


def worst_conditioned(group, modelled, k):
    """The k modelled voxels whose posterior precision is most ill-conditioned."""
    D = group["m"].shape[1]
    r, c = np.triu_indices(D)
    cond = np.empty(len(modelled))
    step = 20000
    for start in range(0, len(modelled), step):
        idx = modelled[start:start + step]
        A = np.zeros((len(idx), D, D))
        A[:, r, c] = group["A_triu"].oindex[idx]
        A[:, c, r] = A[:, r, c]
        cond[start:start + len(idx)] = np.linalg.cond(A)
    order = np.argsort(cond)[::-1][:k]
    return modelled[order], cond[order]


def pick_voxels(group, n, rng):
    """Random modelled voxels plus the cases most likely to break a reader."""
    has = group["has_model"][:]
    be_map = group["be_map"][:]
    modelled = np.flatnonzero(has)
    worst, cond = worst_conditioned(group, modelled, max(1, n // 4))
    print(f"worst cond(A): {cond[0]:.2e} (k={len(worst)})", flush=True)
    flipped = modelled[be_map[modelled, 0] != 0]
    shard = group["has_model"].shards[0]
    per_shard = [modelled[(modelled >= s) & (modelled < s + shard)][[0, -1]]
                 for s in range(0, has.shape[0], shard)
                 if ((modelled >= s) & (modelled < s + shard)).any()]
    picks = np.concatenate([
        rng.choice(modelled, size=n, replace=False),
        rng.choice(flipped, size=min(len(flipped), n // 4), replace=False),
        np.concatenate(per_shard),
        worst,
    ])
    unmodelled = np.flatnonzero(~has)
    return np.unique(picks), unmodelled[:3]


def extract_model(zips, marker, voxel, training_chunk, workdir):
    """Unpack model_<voxel>/ from its training zip; return the model dir."""
    chunk = voxel // training_chunk
    with zipfile.ZipFile(zips / f"Models_n{marker}_c{chunk}.zip") as zf:
        prefix = f"model_{voxel}/"
        members = [n for n in zf.namelist() if n.startswith(prefix)]
        if not members:
            raise KeyError(f"{marker} voxel {voxel} not in chunk {chunk}")
        zf.extractall(workdir, members)
    return os.path.join(workdir, f"model_{voxel}")


def oracle_z(model_dir, rows):
    """PCNtoolkit z for rows [(age, sex, y)] at one voxel."""
    from pcntoolkit import NormativeModel, NormData

    model = NormativeModel.load(model_dir)
    model.saveresults = False
    model.saveplots = False
    frame = pd.DataFrame(rows, columns=["Age", "Sex", "micro"])
    data = NormData.from_dataframe("fixture", frame, covariates=["Age"],
                                   batch_effects=["Sex"], response_vars=["micro"])
    model.compute_zscores(data)
    return data["Z"].sel(response_vars="micro").values.astype(np.float64)


def cases_for(params, rng):
    lo, hi = params["cov_ranges"][0]
    ages = [lo - 15.0, lo, (2 * lo + hi) / 3, (lo + hi) / 2, (lo + 2 * hi) / 3, hi, hi + 9.0,
            float(rng.uniform(lo, hi))]
    out_m, out_s = params["outscaler"][0], params["outscaler"][1]
    cases = []
    for age in ages:
        for sex in SEXES:
            for _ in range(2):
                y = out_m + out_s * rng.normal(0.0, 1.5)
                if y == 0.0:
                    y = out_m
                cases.append({"age": age, "sex": sex, "y": float(y)})
    # Deep in the warp's tails: |ys| of 8 and 12 at a mid-range age.
    for ys in (-12.0, -8.0, 8.0, 12.0):
        cases.append({"age": ages[3], "sex": SEXES[int(ys > 0)], "y": float(out_m + out_s * ys)})
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--zips", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "test/fixtures")
    ap.add_argument("--per-marker", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--markers", nargs="*", help="default: every marker in the store")
    args = ap.parse_args()

    root = zarr.open_group(str(args.store), mode="r")
    attrs = dict(root.attrs)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    workdir = tempfile.mkdtemp(prefix="nm-fixtures-")
    try:
        for marker in args.markers or attrs["markers"]:
            group = root[marker]
            voxels, unmodelled = pick_voxels(group, args.per_marker, rng)
            arrays = {name: group[name].oindex[voxels] for name in PREDICT_ARRAYS}
            records = []
            for k, voxel in enumerate(voxels):
                params = {name: arrays[name][k].tolist() for name in PREDICT_ARRAYS}
                cases = cases_for({"cov_ranges": params["cov_ranges"],
                                   "outscaler": params["outscaler"]}, rng)
                lo, hi = params["cov_ranges"][0]
                rows = [(min(max(c["age"], lo), hi), c["sex"], c["y"]) for c in cases]
                model_dir = extract_model(args.zips, marker, int(voxel),
                                          attrs["training_chunk"], workdir)
                z = oracle_z(model_dir, rows)
                shutil.rmtree(model_dir)
                for case, zk in zip(cases, z, strict=True):
                    case["z"] = float(zk)
                    case["clamped"] = not (lo <= case["age"] <= hi)
                records.append({"voxel": int(voxel), "params": params, "cases": cases})
                print(f"{marker} voxel {voxel}: {len(cases)} cases", flush=True)
            doc = {
                "source": {
                    "store_family": attrs["family"],
                    "nmzarr_format_version": attrs["nmzarr_format_version"],
                    "mask_sha256": attrs["mask_sha256"],
                    "oracle": "pcntoolkit NormativeModel.compute_zscores on the training zip model",
                    "seed": args.seed,
                },
                "marker": marker,
                "be_level_columns": list(group.attrs["be_level_columns"]),
                "unmodelled_voxels": unmodelled.tolist(),
                "voxels": records,
            }
            path = args.out / f"predict-{marker}.json"
            path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
            print(f"wrote {path} ({len(records)} voxels)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
