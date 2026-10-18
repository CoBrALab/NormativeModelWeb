"""Prepare the model store for publication; does not upload anything.

Copies an nmzarr store to --out and writes models/normative-dwi.manifest.json
(scientific-assets-v1: one entry per file, with bytes and SHA-256).

Upload --out to the Hugging Face dataset, then rerun with --base-url set to
the pinned resolve URL (https://huggingface.co/datasets/<repo>/resolve/<commit>/<path>/)
and set DEFAULT_STORE_URL in apps/normative-dwi/src/config.ts to the same URL.

The store derives from UK Biobank data: publish only once
redistribution is cleared.

Usage (reference env: /scratch/parent41/voxel_nm/.venv_pcnGabe):
  python -I tools/prepare_store_release.py --store STORE --out OUT [--base-url URL]
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import zarr

ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default=None, help="pinned URL of --out once uploaded")
    ap.add_argument("--manifest", type=Path, default=ROOT / "models/normative-dwi.manifest.json")
    args = ap.parse_args()

    attrs = dict(zarr.open_group(str(args.store), mode="r").attrs)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; remove it first")
    shutil.copytree(args.store, args.out)

    assets = []
    for path in sorted(p for p in args.out.rglob("*") if p.is_file()):
        rel = path.relative_to(args.out).as_posix()
        assets.append({"filename": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "schema_version": 1,
        "app": "normative-dwi",
        "base_url": args.base_url,
        "license": "NOASSERTION",
        "store": {
            "family": attrs["family"],
            "nmzarr_format_version": attrs["nmzarr_format_version"],
            "mask_sha256": attrs["mask_sha256"],
            "source_ptk_version": attrs["source_ptk_version"],
        },
        "assets": assets,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(a["bytes"] for a in assets)
    print(f"{args.out}: {len(assets)} files, {total / 1e6:.1f} MB; manifest {args.manifest}")


if __name__ == "__main__":
    main()
