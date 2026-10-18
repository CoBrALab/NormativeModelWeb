"""A committed, data-free reference for the z-score kernel.

Fits a few voxel models with PCNtoolkit itself on SIMULATED data, with the
configuration of the UK Biobank models (BLR, sinh-arcsinh warp with
reparametrisation, heteroskedastic noise on age, cubic B-spline in age with 4
knots and 0.2 expansion, sex as a fixed batch effect). It then scores cases
with PCNtoolkit's compute_zscores and writes test/reference/synthetic-pcn.json:
the parameters as the nmzarr store holds them, the cases, and PCNtoolkit's z.

Nothing here comes from UK Biobank, so the output is committed, and CI checks
the browser kernel against an independent implementation on every run. Half
of the voxels see Sex "1" first, so PCNtoolkit maps it to the first design
column and the be_map is swapped, as in about 7.5% of real FA voxels.

Ages outside the training range are clamped before scoring, as the browser
does (see make_fixtures.py); the cases include clamped ages and |ys| up to 10.

Usage (reference env: /scratch/parent41/voxel_nm/.venv_pcnGabe):
  python -I apps/normative-dwi/tools/make_synthetic_reference.py
"""

import argparse
import importlib.metadata
import json
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from make_fixtures import oracle_z

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
SEXES = ["0", "1"]
LEVEL_COLUMNS = ["Sex=0", "Sex=1"]


def fit(save_dir, frame):
    from pcntoolkit import BLR, BsplineBasisFunction, LinearBasisFunction, NormativeModel, NormData

    template = BLR(
        name="blr", fixed_effect=True, heteroskedastic=True, fixed_effect_var=False,
        warp_name="warpsinharcsinh", warp_reparam=True,
        basis_function_mean=BsplineBasisFunction(basis_column=0, degree=3, nknots=4, left_expand=0.2, right_expand=0.2),
        basis_function_var=LinearBasisFunction(basis_column=0),
        n_iter=300, tol=1e-9, ard=False, optimizer="l-bfgs-b", l_bfgs_b_l=0.1, l_bfgs_b_epsilon=0.1, l_bfgs_b_norm="l2",
        hyp0=None, is_fitted=False, is_from_dict=False,
    )
    model = NormativeModel(template_regression_model=template, savemodel=True, evaluate_model=False, saveresults=False,
                           saveplots=False, inscaler="standardize", outscaler="standardize", save_dir=save_dir)
    data = NormData.from_dataframe("train", frame, covariates=["Age"], batch_effects=["Sex"], response_vars=["micro"])
    model.fit(data)


def params_from_model(model_dir):
    """The predict arrays as the nmzarr store holds them (cf. nmzarr read_pcn14.voxel_record)."""
    norm = json.loads((Path(model_dir) / "model/normative_model.json").read_text())
    reg_doc = json.loads((Path(model_dir) / "model/micro/regression_model.json").read_text())
    reg, out = reg_doc["model"], reg_doc["outscaler"]
    A = np.asarray(reg["A"], dtype=np.float64)
    D = A.shape[0]
    scaler = norm["inscalers"]["Age"]
    be = {f"Sex={level}": col for level, col in reg["be_maps"]["Sex"].items()}
    ranges = norm["covariate_ranges"]["Age"]
    return {
        "has_model": True,
        "m": [float(v) for v in reg["m"]],
        "A_triu": [float(v) for v in A[np.triu_indices(D)]],
        "hyp": [float(v) for v in reg["hyp"]],
        "mean_knots": [float(v) for v in reg["basis_function_mean"]["knots"]],
        "inscaler": [[float(scaler[k]) for k in ("m", "s", "min", "max")]],
        "outscaler": [float(out[k]) for k in ("m", "s", "min", "max")],
        "be_map": [int(be[c]) for c in LEVEL_COLUMNS],
        "cov_ranges": [[float(ranges["min"]), float(ranges["max"])]],
    }


def simulate(rng, n, sex_first):
    """A marker-like response: smooth age trend, sex offset, age-dependent skewed noise."""
    age = rng.uniform(45, 81, n)
    sex = rng.choice(SEXES, n)
    sex[0] = sex_first
    base = 0.45 + 0.03 * rng.normal()
    slope = -0.004 + 0.001 * rng.normal()
    curve = 0.00015 * rng.normal()
    offset = np.where(sex == "1", 0.02 * rng.normal(), 0.0)
    noise = np.sinh(0.35 * rng.normal(size=n) + 0.15) * (0.03 + 0.0006 * (age - 45))
    y = base + slope * (age - 60) + curve * (age - 60) ** 2 + offset + noise
    return pd.DataFrame({"Age": age, "Sex": sex, "micro": y})


def cases_for(params, rng):
    lo, hi = params["cov_ranges"][0]
    out_m, out_s = params["outscaler"][0], params["outscaler"][1]
    cases = []
    for age in (lo - 12.0, lo, 55.0, 63.3, 72.0, hi, hi + 7.0):
        for sex in SEXES:
            cases.append({"age": age, "sex": sex, "y": float(out_m + out_s * rng.normal(0, 1.5))})
    for ys in (-10.0, -6.0, 6.0, 10.0):
        cases.append({"age": 60.0, "sex": SEXES[int(ys > 0)], "y": float(out_m + out_s * ys)})
    return cases


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--voxels", type=int, default=8)
    ap.add_argument("--subjects", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", type=Path, default=HERE.parent / "test/reference/synthetic-pcn.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    work = Path(tempfile.mkdtemp(prefix="nm-synthetic-"))
    records = []
    try:
        for v in range(args.voxels):
            model_dir = work / f"voxel_{v}"
            fit(str(model_dir), simulate(rng, args.subjects, SEXES[v % 2]))
            params = params_from_model(model_dir)
            cases = cases_for(params, rng)
            lo, hi = params["cov_ranges"][0]
            rows = [(min(max(c["age"], lo), hi), c["sex"], c["y"]) for c in cases]
            z = oracle_z(str(model_dir), rows)
            for case, zk in zip(cases, z, strict=True):
                case["z"] = float(zk)
                case["clamped"] = not (lo <= case["age"] <= hi)
            records.append({"voxel": v, "params": params, "cases": cases})
            print(f"voxel {v}: be_map {params['be_map']}, {len(cases)} cases", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    doc = {
        "source": {
            "data": "simulated; no UK Biobank or subject data",
            "oracle": "pcntoolkit NormativeModel.compute_zscores on models PCNtoolkit fitted to the simulated data",
            "pcntoolkit": importlib.metadata.version("pcntoolkit"),
            "pcntoolkit_source": "git+https://github.com/gdevenyi/PCNToolkit.git@4219457a92a9313f48482158d329d7ff59edd082",
            "seed": args.seed,
            "command": "python -I apps/normative-dwi/tools/make_synthetic_reference.py",
        },
        "marker": "synthetic",
        "be_level_columns": LEVEL_COLUMNS,
        "voxels": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
