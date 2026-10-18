"""Full-volume reference z-maps for the end-to-end test (local data only).

Takes one subject's template-space maps as proc_nm writes them
(maps_UKB_<stub>.mnc, store/MINC axis order), and writes to --out:

  input_<MARKER>.nii.gz      the map as an ordinary x-fastest NIfTI, which is
                             what a user would upload
  reference_<MARKER>.nii.gz  its z-map, computed here with numpy straight
                             from the store arrays
  reference.json             the covariates used

The z formula is written independently of the browser kernel (Cox-de Boor
recursion, batched linear solves) and matches PCNtoolkit on the committed
fixtures; the end-to-end test therefore checks the browser's image I/O and
voxel mapping on a real brain. Subject data: never commit the output.

Usage (reference env: /scratch/parent41/voxel_nm/.venv_pcnGabe):
  python -I tools/make_reference_zmap.py --store STORE --maps DIR --age 72 --sex female --out OUT
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr

STUBS = {"FA": "dti_FA", "MD": "dti_MD", "ICVF": "noddi_ICVF", "ISOVF": "noddi_ISOVF", "OD": "noddi_OD"}


def bspline_basis(x, knots, degree=3):
    """B_i(x) for every voxel by the Cox-de Boor recursion; x (n,), knots (n, K)."""
    n, K = knots.shape
    B = ((knots[:, :-1] <= x[:, None]) & (x[:, None] < knots[:, 1:])).astype(np.float64)
    for d in range(1, degree + 1):
        nxt = np.zeros((n, K - d - 1))
        for i in range(K - d - 1):
            den1 = knots[:, i + d] - knots[:, i]
            den2 = knots[:, i + d + 1] - knots[:, i + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(den1 > 0, (x - knots[:, i]) / den1, 0.0) * B[:, i]
                t2 = np.where(den2 > 0, (knots[:, i + d + 1] - x) / den2, 0.0) * B[:, i + 1]
            nxt[:, i] = t1 + t2
        B = nxt
    return B


def zscores(group, y, age, sex_level):
    has = group["has_model"][:]
    cols = list(group.attrs["be_level_columns"])
    valid = has & np.isfinite(y) & (y != 0)
    idx = np.flatnonzero(valid)
    get = lambda name: group[name].oindex[idx]
    m, a_triu, hyp, knots = get("m"), get("A_triu"), get("hyp"), get("mean_knots")
    ins, outs, be_map, ranges = get("inscaler")[:, 0], get("outscaler"), get("be_map"), get("cov_ranges")[:, 0]
    D = m.shape[1]

    a = np.clip(age, ranges[:, 0], ranges[:, 1])
    x = (a - ins[:, 0]) / ins[:, 1]
    if np.any((x < knots[:, 3]) | (x >= knots[:, -4])):
        raise SystemExit("an age falls outside a voxel's knot span; this script does not extrapolate")
    basis = bspline_basis(x, knots)
    onehot = np.zeros((len(idx), be_map.shape[1]))
    onehot[np.arange(len(idx)), be_map[:, cols.index(sex_level)]] = 1
    phi = np.concatenate([basis, np.ones((len(idx), 1)), onehot], axis=1)

    A = np.zeros((len(idx), D, D))
    r, c = np.triu_indices(D)
    A[:, r, c] = a_triu
    A[:, c, r] = a_triu
    quad = np.einsum("nd,nd->n", phi, np.linalg.solve(A, phi[..., None])[..., 0])

    b = np.exp(hyp[:, 3])
    warp_a = -hyp[:, 2] * b
    beta = np.exp(x * hyp[:, 0] + hyp[:, 1]) / b**2
    s2 = 1 / beta + quad
    mu = np.einsum("nd,nd->n", phi, m)
    ys = (y[idx] - outs[:, 0]) / outs[:, 1]
    z = np.zeros_like(y)
    z[idx] = (np.sinh(b * np.arcsinh(ys) - warp_a) - mu) / np.sqrt(s2)
    return z, valid


def to_nifti(store_order, store_affine):
    """Store/MINC-order volume -> x-fastest NIfTI on the same grid."""
    perm = [int(np.flatnonzero(store_affine[r, :3])[0]) for r in range(3)]  # world axis r <- array axis perm[r]
    data = np.transpose(store_order, perm)
    affine = np.eye(4)
    for r in range(3):
        affine[r, r] = store_affine[r, perm[r]]
        affine[r, 3] = store_affine[r, 3]
    return nib.Nifti1Image(data.astype(np.float32), affine)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--maps", type=Path, required=True, help="folder with maps_UKB_<stub>.mnc")
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--sex", choices=["female", "male"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    root = zarr.open_group(str(args.store), mode="r")
    mask = root["mask"][:] > 0
    store_affine = root["affine"][:]
    ijk = root["ijk"][:]
    args.out.mkdir(parents=True, exist_ok=True)
    level = "Sex=0" if args.sex == "female" else "Sex=1"
    for marker, stub in STUBS.items():
        path = args.maps / f"maps_UKB_{stub}.mnc"
        if not path.exists():
            print(f"skip {marker}: no {path.name}")
            continue
        img = nib.load(str(path))
        if img.shape != mask.shape or not np.allclose(img.affine, store_affine, atol=1e-3):
            raise SystemExit(f"{path} is not on the store grid")
        vol = np.asarray(img.dataobj, dtype=np.float64)
        y = vol[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        z, valid = zscores(root[marker], y, args.age, level)
        zvol = np.zeros(mask.shape)
        zvol[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.where(valid, z, 0)
        nib.save(to_nifti(vol, store_affine), str(args.out / f"input_{marker}.nii.gz"))
        nib.save(to_nifti(zvol, store_affine), str(args.out / f"reference_{marker}.nii.gz"))
        print(f"{marker}: {valid.sum()} voxels scored, |z| median {np.median(np.abs(z[valid])):.2f}")
    (args.out / "reference.json").write_text(json.dumps({"age": args.age, "sex": args.sex}) + "\n")


if __name__ == "__main__":
    main()
