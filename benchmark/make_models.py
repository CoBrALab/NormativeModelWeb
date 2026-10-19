"""Synthetic training archives for one marker, laid out like proc_nm's.

Writes to --out:

  Models_n<MARKER>_c<CHUNK>.zip   PCNtoolkit voxel models, TRAINING_CHUNK voxels
                                  per archive, model_<global voxel>/model/{
                                  normative_model.json, micro/regression_model.json},
                                  stored uncompressed, as training writes them
  mask.nii.gz                     the voxel grid (an ellipsoid on the UK Biobank
                                  2 mm grid, 97 x 115 x 97)
  manifest.json                   grid sizes, modelled voxels, the timing sample

Every model is a real PCNtoolkit fit, or a perturbed copy of one. The script
fits --prototypes voxel models to simulated data with the design of the UK
Biobank models (make_synthetic_reference.fit: BLR, sinh-arcsinh warp, noise
linear in age, cubic B-spline in age, Sex as a batch effect). It then fills
every modelled voxel with a copy of a random prototype, perturbed so that no
two voxels hold the same numbers (see perturb()). With --prototypes at or
above the number of modelled voxels, every voxel is fitted.

The grid, the modelled voxels and the timing sample depend only on --seed and
the sizes, so two runs in different PCNtoolkit environments describe the same
voxels.

Usage:
  python -I benchmark/make_models.py --out DIR [--n-vox 304327] [--prototypes 512]
"""

import argparse
import contextlib
import copy
import importlib.metadata
import io
import json
import math
import shutil
import sys
import tempfile
import time
import warnings
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "apps/normative-dwi/tools"))
from make_synthetic_reference import SEXES, fit, simulate

warnings.filterwarnings("ignore")

GRID = (97, 115, 97)
# UK Biobank 2 mm template affine (voxel -> mm).
AFFINE = np.array([[-2.0, 0, 0, 96], [0, 2.0, 0, -132], [0, 0, 2.0, -78], [0, 0, 0, 1]])
TRAINING_CHUNK = (
    10000  # voxels per archive, as in proc_nm and nmzarr layout.TRAINING_CHUNK
)


def make_grid(n_vox, modelled_fraction):
    """Mask of exactly n_vox voxels; the outer shell is left unmodelled.

    In the real models the unmodelled voxels (too few valid observations) sit
    at the edge of the dilated mask, so the shell is the closest simple stand-in.
    """
    idx = np.indices(GRID, dtype=np.float64)
    centre = (np.array(GRID) - 1) / 2
    radii = np.array(GRID) * 0.42
    r2 = sum(((idx[d] - centre[d]) / radii[d]) ** 2 for d in range(3)).ravel()
    order = np.argsort(r2, kind="stable")
    inside = np.zeros(r2.size, dtype=bool)
    inside[order[:n_vox]] = True
    mask = inside.reshape(GRID)
    # Flat voxel j = j-th mask voxel in C order (nmzarr spatial.VOXEL_ORDER).
    flat = np.flatnonzero(inside)
    rank = np.empty(r2.size, dtype=np.int64)
    rank[order] = np.arange(r2.size)
    modelled = rank[flat] < round(modelled_fraction * n_vox)
    return mask, modelled


def timing_sample(modelled, per_chunk):
    """per_chunk modelled voxels from every archive, evenly spaced."""
    sample = []
    for lo in range(0, modelled.size, TRAINING_CHUNK):
        ids = lo + np.flatnonzero(modelled[lo : lo + TRAINING_CHUNK])
        if ids.size:
            sample += ids[
                np.linspace(0, ids.size - 1, min(per_chunk, ids.size))
                .round()
                .astype(int)
            ].tolist()
    return sorted(set(sample))


def dumps_pcn14(o, level=0):
    """json.dumps as PCNtoolkit 1.4 writes models: indent 4, lists of numbers on one line."""
    pad = "    " * (level + 1)
    if isinstance(o, dict) and o:
        items = ",\n".join(
            f"{pad}{json.dumps(k)}: {dumps_pcn14(v, level + 1)}" for k, v in o.items()
        )
        return "{\n" + items + "\n" + "    " * level + "}"
    if (
        isinstance(o, list)
        and o
        and not all(
            isinstance(x, (int, float, list)) and not isinstance(x, bool) for x in o
        )
    ):
        return (
            "[\n"
            + ",\n".join(pad + dumps_pcn14(x, level + 1) for x in o)
            + "\n"
            + "    " * level
            + "]"
        )
    return json.dumps(o)


def dumps_indent4(o):
    return json.dumps(o, indent=4)


def serializer_for(texts):
    """The serializer that reproduces PCNtoolkit's own files byte for byte, so copies are the same size."""
    for dumps in (dumps_pcn14, dumps_indent4):
        if all(dumps(json.loads(t)) == t for t in texts):
            return dumps
    raise SystemExit(
        "Error: cannot reproduce PCNtoolkit's JSON layout; copies would not be the size of real models"
    )


def perturb(norm, reg_doc, rng, save_dir):
    """A different but valid model: same design and JSON layout, new numbers.

    - Age scaler: new mean and sd, with the knots, the basis ranges and the
      noise hyperparameters re-expressed so the model is unchanged in years.
    - Weights m scaled per entry; precision A -> S A S (S diagonal, positive),
      which keeps it symmetric positive definite.
    - Warp hyperparameters shifted, gamma kept equal to hyp[2:4].
    - Response scaler: new mean and sd, min and max re-expressed.
    Sigma_a and Lambda_a (PCNtoolkit 1.3.0 only) are the training prior and
    are not read at prediction; they are left as they are.
    """
    norm, reg_doc = copy.deepcopy(norm), copy.deepcopy(reg_doc)
    reg = reg_doc["model"]

    sc = norm["inscalers"]["Age"]
    m0, s0 = sc["m"], sc["s"]
    m1, s1 = m0 + 0.5 * rng.normal(), s0 * (1 + 0.01 * rng.normal())
    sc["m"], sc["s"] = m1, s1

    def remap(x):
        return ((m0 + s0 * x) - m1) / s1

    bm, bv = reg["basis_function_mean"], reg["basis_function_var"]
    bm["knots"] = [remap(k) for k in bm["knots"]]
    for basis in (bm, bv):
        basis["min"], basis["max"] = remap(basis["min"]), remap(basis["max"])
    # beta = exp(x h0 + h1) in the old x; the same function of age in the new x.
    hyp = np.asarray(reg["hyp"], dtype=np.float64)
    hyp[1] += hyp[0] * (m1 - m0) / s0
    hyp[0] *= s1 / s0

    D = len(reg["m"])
    reg["m"] = (np.asarray(reg["m"]) * (1 + 0.05 * rng.normal(size=D))).tolist()
    s = np.exp(0.05 * rng.normal(size=D))
    reg["A"] = (np.asarray(reg["A"]) * np.outer(s, s)).tolist()
    tied_gamma = reg.get("gamma") == reg["hyp"][2 : 2 + len(reg.get("gamma") or [])]
    hyp[2:] += 0.02 * rng.normal(size=hyp.size - 2)
    reg["hyp"] = hyp.tolist()
    if tied_gamma and reg.get("gamma"):
        reg["gamma"] = hyp[2 : 2 + len(reg["gamma"])].tolist()
    reg["nlZ"] = reg["nlZ"] * (1 + 0.01 * rng.normal())

    out = reg_doc["outscaler"]
    om, os_ = out["m"], out["s"]
    om1, os1 = om * (1 + 0.02 * rng.normal()), os_ * (1 + 0.02 * rng.normal())
    out["min"] = om1 + (out["min"] - om) * os1 / os_
    out["max"] = om1 + (out["max"] - om) * os1 / os_
    out["m"], out["s"] = om1, os1

    norm["save_dir"] = save_dir
    return norm, reg_doc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--marker", default="FA")
    ap.add_argument(
        "--n-vox",
        type=int,
        default=304327,
        help="mask voxels (default: the real FA store)",
    )
    ap.add_argument("--modelled-fraction", type=float, default=241373 / 304327)
    ap.add_argument(
        "--prototypes", type=int, default=512, help="voxel models fitted by PCNtoolkit"
    )
    ap.add_argument(
        "--train-subjects", type=int, default=1000, help="simulated subjects per fit"
    )
    ap.add_argument(
        "--sample-per-chunk",
        type=int,
        default=20,
        help="voxels per archive in the timing sample",
    )
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()

    import pcntoolkit.util.output

    pcntoolkit.util.output.Output.set_show_messages(False)

    if args.out.exists():
        raise SystemExit(f"Error: {args.out} exists; remove it first")
    args.out.mkdir(parents=True)
    t_start = time.perf_counter()

    mask, modelled = make_grid(args.n_vox, args.modelled_fraction)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), AFFINE), args.out / "mask.nii.gz")
    n_models = int(modelled.sum())
    n_proto = min(args.prototypes, n_models)

    # Independent streams: changing --prototypes does not move the grid or the sample.
    rng_data = np.random.default_rng([args.seed, 1])
    rng_copy = np.random.default_rng([args.seed, 2])

    work = Path(tempfile.mkdtemp(prefix=".fit-", dir=args.out))  # under --out, not /tmp
    protos = []
    t_fit = time.perf_counter()
    try:
        for p in range(n_proto):
            model_dir = work / f"p{p}"
            with contextlib.redirect_stdout(io.StringIO()):
                fit(
                    str(model_dir),
                    simulate(rng_data, args.train_subjects, SEXES[p % 2]),
                )
            texts = [
                (model_dir / "model/normative_model.json").read_text(),
                (model_dir / "model/micro/regression_model.json").read_text(),
            ]
            if p == 0:
                dumps = serializer_for(texts)
            protos.append(tuple(json.loads(t) for t in texts))
            shutil.rmtree(model_dir, ignore_errors=True)
            if (p + 1) % 64 == 0 or p + 1 == n_proto:
                print(
                    f"fitted {p + 1}/{n_proto} prototypes ({time.perf_counter() - t_fit:.0f}s)",
                    flush=True,
                )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    fit_s = time.perf_counter() - t_fit

    every_voxel_fitted = n_proto == n_models
    n_chunks = math.ceil(args.n_vox / TRAINING_CHUNK)
    model_ids = np.flatnonzero(modelled)
    for c in range(n_chunks):
        lo, hi = c * TRAINING_CHUNK, min((c + 1) * TRAINING_CHUNK, args.n_vox)
        path = args.out / f"Models_n{args.marker}_c{c}.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
            for v in model_ids[(model_ids >= lo) & (model_ids < hi)]:
                rank = int(np.searchsorted(model_ids, v))
                if every_voxel_fitted:
                    norm, reg_doc = (
                        copy.deepcopy(protos[rank][0]),
                        copy.deepcopy(protos[rank][1]),
                    )
                    norm["save_dir"] = (
                        f"/dev/shm/bench/Models_n{args.marker}_c{c}/model_{v}"
                    )
                else:
                    norm, reg_doc = perturb(
                        *protos[rng_copy.integers(n_proto)],
                        rng_copy,
                        f"/dev/shm/bench/Models_n{args.marker}_c{c}/model_{v}",
                    )
                zf.writestr(
                    f"model_{v}/model/micro/regression_model.json", dumps(reg_doc)
                )
                zf.writestr(f"model_{v}/model/normative_model.json", dumps(norm))
        print(f"wrote {path.name}", flush=True)

    manifest = {
        "marker": args.marker,
        "grid": list(GRID),
        "n_vox": args.n_vox,
        "n_models": n_models,
        "n_chunks": n_chunks,
        "training_chunk": TRAINING_CHUNK,
        "prototypes": n_proto,
        "every_voxel_fitted": every_voxel_fitted,
        "train_subjects": args.train_subjects,
        "seed": args.seed,
        "pcntoolkit": importlib.metadata.version("pcntoolkit"),
        "design_D": len(protos[0][1]["model"]["m"]),
        "fit_s_per_model": fit_s / n_proto,
        "build_s": time.perf_counter() - t_start,
        "sample": timing_sample(modelled, args.sample_per_chunk),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(
        f"done: {n_models} models over {args.n_vox} voxels in {n_chunks} archives "
        f"({manifest['build_s']:.0f}s)"
    )


if __name__ == "__main__":
    main()
