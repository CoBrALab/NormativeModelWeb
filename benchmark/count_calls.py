"""Python function calls PCNtoolkit makes per voxel (cProfile), for RESULTS.md.

Counts the calls of NormData.from_dataframe, compute_zscores and predict for one
subject, averaged over the first 20 modelled voxels of the first archive.

Usage (in a PCNtoolkit environment):
  python -I benchmark/count_calls.py --models DIR --out results/calls-<label>.json [--scratch DIR]
"""

import argparse
import contextlib
import cProfile
import io
import json
import pstats
import shutil
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
N_VOXELS = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args()

    import pcntoolkit.util.output
    from pcntoolkit import NormativeModel, NormData

    pcntoolkit.util.output.Output.set_show_messages(False)
    scratch = Path(tempfile.mkdtemp(prefix="nm-bench-calls-", dir=args.scratch))
    frame = pd.DataFrame({"uid": ["s0"], "Age": [60.0], "Sex": ["0"], "micro": [0.45]})

    def make_data():
        return NormData.from_dataframe(
            name="b",
            dataframe=frame,
            covariates=["Age"],
            batch_effects=["Sex"],
            response_vars=["micro"],
            subject_ids="uid",
        )

    def calls(fn):
        profile = cProfile.Profile()
        with contextlib.redirect_stdout(io.StringIO()):
            profile.enable()
            fn()
            profile.disable()
        return pstats.Stats(profile).total_calls

    try:
        archive = min(args.models.glob("Models_n*_c0.zip"))
        with zipfile.ZipFile(archive) as zf:
            voxels = sorted({n.split("/")[0] for n in zf.namelist()})[:N_VOXELS]
            zf.extractall(
                scratch, [n for n in zf.namelist() if n.split("/")[0] in voxels]
            )
        totals = {"NormData.from_dataframe": 0, "compute_zscores": 0, "predict": 0}
        for voxel in voxels:
            with contextlib.redirect_stdout(io.StringIO()):
                model = NormativeModel.load(str(scratch / voxel))
                model.saveresults = model.saveplots = model.savemodel = False
                model.save_dir = tempfile.mkdtemp(dir=scratch)
            totals["NormData.from_dataframe"] += calls(make_data)
            for name in ("compute_zscores", "predict"):
                data = make_data()
                totals[name] += calls(
                    lambda name=name, data=data, model=model: getattr(model, name)(data)
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    doc = {
        "pcntoolkit": pcntoolkit_version(),
        "voxels": len(voxels),
        "calls_per_voxel": {k: v // len(voxels) for k, v in totals.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["calls_per_voxel"]))


def pcntoolkit_version():
    import importlib.metadata

    return importlib.metadata.version("pcntoolkit")


if __name__ == "__main__":
    sys.exit(main())
