"""Bytes and objects: the training archives against the nmzarr store.

Standard library only. For the archives: files, bytes, JSON entries, and what
one voxel's JSON is made of. For the store: bytes and objects per array, the
part the app downloads (the predict arrays), and their size once decoded.

Usage:
  python -I benchmark/measure_storage.py --archives pcn14=DIR [--archives pcn13=DIR] --store STORE --out results/storage.json
"""

import argparse
import json
import zipfile
from pathlib import Path

# The arrays the app reads (apps/normative-dwi/src/model-store/index.ts, FIELDS).
PREDICT_ARRAYS = [
    "has_model",
    "m",
    "A_triu",
    "hyp",
    "mean_knots",
    "inscaler",
    "outscaler",
    "be_map",
    "cov_ranges",
]
DTYPE_BYTES = {
    "bool": 1,
    "int16": 2,
    "int32": 4,
    "uint8": 1,
    "float64": 8,
    "float32": 4,
}


def files_under(path):
    return [p for p in Path(path).rglob("*") if p.is_file()]


def json_anatomy(zip_path):
    """One voxel's two JSON files: total, the numbers prediction needs, the rest."""
    with zipfile.ZipFile(zip_path) as zf:
        first = next(n for n in zf.namelist() if n.endswith("regression_model.json"))
        voxel = first.split("/")[0]
        reg_text = zf.read(first)
        norm_text = zf.read(f"{voxel}/model/normative_model.json")
    reg_doc = json.loads(reg_text)
    reg = reg_doc["model"]
    D = len(reg["m"])
    out = reg_doc["outscaler"]
    needed = [
        reg["A"],
        reg["m"],
        reg["hyp"],
        reg["basis_function_mean"]["knots"],
        [out[k] for k in ("m", "s", "min", "max")],
    ]
    # The numbers alone, written as compactly as JSON allows.
    needed_text = sum(len(json.dumps(v, separators=(",", ":"))) for v in needed)
    # Store row: m D + A_triu D(D+1)/2 + hyp 5 + knots + inscaler 4 + outscaler 4 + cov_ranges 2 float64, be_map int16s.
    n_float = (
        D
        + D * (D + 1) // 2
        + len(reg["hyp"])
        + len(reg["basis_function_mean"]["knots"])
        + 4
        + 4
        + 2
    )
    total = len(norm_text) + len(reg_text)
    return {
        "D": D,
        "json_bytes": total,
        "json_bytes_predict_fields": needed_text,
        "json_bytes_other": total - needed_text,
        "a_has_lambda_sigma": "Lambda_a" in reg,
        "float64_values_predict": n_float,
        "binary_bytes_predict": 8 * n_float,
    }


def archives(path):
    zips = sorted(Path(path).glob("Models_n*_c*.zip"))
    n_entries = n_raw = 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            infos = zf.infolist()
            n_entries += len(infos)
            n_raw += sum(i.file_size for i in infos)
    n_models = n_entries // 2
    total = sum(z.stat().st_size for z in zips)
    return {
        "files": len(zips),
        "bytes": total,
        "json_entries": n_entries,
        "json_bytes_uncompressed": n_raw,
        "models": n_models,
        "bytes_per_model": total / n_models,
        "voxel_json": json_anatomy(zips[0]),
    }


def array_info(array_dir):
    meta = json.loads((array_dir / "zarr.json").read_text())
    data = [p for p in files_under(array_dir) if p.name != "zarr.json"]
    n = 1
    for d in meta["shape"]:
        n *= d
    return {
        "shape": meta["shape"],
        "dtype": meta["data_type"],
        "objects": len(data),
        "bytes": sum(p.stat().st_size for p in data),
        "decoded_bytes": n * DTYPE_BYTES[meta["data_type"]],
    }


def store(path):
    path = Path(path)
    root = json.loads((path / "zarr.json").read_text())
    markers = root["attributes"]["markers"]
    out = {
        "bytes": sum(p.stat().st_size for p in files_under(path)),
        "objects": len(files_under(path)),
        "shared": {name: array_info(path / name) for name in ("mask", "affine", "ijk")},
        "markers": {},
    }
    for marker in markers:
        arrays = {
            d.name: array_info(d)
            for d in sorted((path / marker).iterdir())
            if d.is_dir()
        }
        predict = {k: arrays[k] for k in PREDICT_ARRAYS}
        out["markers"][marker] = {
            "bytes": sum(a["bytes"] for a in arrays.values()),
            "decoded_bytes": sum(a["decoded_bytes"] for a in arrays.values()),
            "arrays": arrays,
            "predict_bytes": sum(a["bytes"] for a in predict.values()),
            "predict_objects": sum(a["objects"] for a in predict.values()),
            "predict_decoded_bytes": sum(a["decoded_bytes"] for a in predict.values()),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--archives", action="append", default=[], help="LABEL=DIR of Models_n*_c*.zip"
    )
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    doc = {
        "archives": {
            label: archives(d) for label, d in (a.split("=", 1) for a in args.archives)
        },
        "store": store(args.store),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    for label, a in doc["archives"].items():
        print(
            f"archives {label}: {a['files']} files, {a['bytes'] / 1e6:.0f} MB, {a['bytes_per_model']:.0f} B/model"
        )
    for marker, m in doc["store"]["markers"].items():
        print(
            f"store {marker}: {m['bytes'] / 1e6:.1f} MB on disk, app reads {m['predict_bytes'] / 1e6:.1f} MB "
            f"in {m['predict_objects']} objects, {m['predict_decoded_bytes'] / 1e6:.1f} MB decoded"
        )


if __name__ == "__main__":
    main()
