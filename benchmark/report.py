"""Turn the result JSONs into results/summary.md (the tables RESULTS.md quotes).

Usage: python -I benchmark/report.py [--results benchmark/results]
"""

import argparse
import json
from pathlib import Path


def dur(s):
    if s >= 3600:
        return f"{s / 3600:.1f} h"
    if s >= 120:
        return f"{s / 60:.0f} min"
    if s >= 1:
        return f"{s:.1f} s"
    return f"{s * 1e3:.0f} ms"


def size(b):
    return (
        f"{b / 1e9:.2f} GB"
        if b >= 1e9
        else f"{b / 1e6:.0f} MB"
        if b >= 1e6
        else f"{b / 1e3:.1f} kB"
    )


def gain(old, new):
    g = old / new
    return f"{g:,.0f}×" if g >= 10 else f"{g:.1f}×"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--results", type=Path, default=Path(__file__).resolve().parent / "results"
    )
    args = ap.parse_args()
    r = args.results
    o13 = json.loads((r / "old-pcn13.json").read_text())
    o14 = json.loads((r / "old-pcn14.json").read_text())
    new = json.loads((r / "new.json").read_text())
    st = json.loads((r / "storage.json").read_text())
    env = json.loads((r / "environment.json").read_text())
    man = json.loads((r / "manifest-pcn14.json").read_text())

    n_models, n_vox = man["n_models"], man["n_vox"]
    cohorts = sorted({c["subjects"] for c in o14["configs"].values()})
    big = max(cohorts)
    c13, c14, cn = o13["configs"], o14["configs"], new["configs"]

    # Parity on the timed voxels, both sexes: subject 0 is Sex "0", subject 1 is Sex "1".
    def worst(old):
        return max(
            abs(zo - zk)
            for v, zs in old.items()
            for zo, zk in zip(zs, new["z_subjects01"][v], strict=True)
        )

    parity14 = worst(c14[f"zscores/{big}"]["z_subjects01"])
    parity13 = worst(c13[f"zscores/{big}"]["z_subjects01"])
    n_timed = len(new["z_subjects01"])

    L = []
    w = L.append
    w(f"# Benchmark summary ({env['date']})\n")
    w(
        f"Synthetic marker: {n_models:,} modelled voxels of {n_vox:,} ({man['n_chunks']} archives); "
        f"{man['prototypes']} PCNtoolkit fits per toolkit, perturbed copies elsewhere. "
        f"Old path timed on {o14['voxels_timed']} voxels (every archive) and scaled to the marker; "
        f"unpacking timed on every archive; new path measured on the whole marker. One CPU core.\n"
    )
    w(
        f"Machine: {env['cpu']}, {env['os']}; Node {env['node']}; Python {o13['python']} (PCNtoolkit {o13['pcntoolkit']}), "
        f"Python {o14['python']} (PCNtoolkit {o14['pcntoolkit']}).\n"
    )
    w(
        f"Same models, same subjects (one female, one male): the fork's `compute_zscores` and the app kernel agree to "
        f"max |Δz| = {parity14:.1e} on the {n_timed} timed voxels. PCNtoolkit 1.3.0 scores its own D = 10 fits, which "
        f"the kernel cannot score: its z differs from the kernel's on the same voxels by up to {parity13:.1f}, so the "
        f"1.3.0 rows compare different fits.\n"
    )

    def gain_range(old, nn):
        lo, hi = old / nn["marker_s"], old / nn["marker_s_warm"]
        return (
            f"{gain(old, nn['marker_s'])}–{gain(old, nn['marker_s_warm'])}"
            if abs(hi - lo) > 0.02 * lo
            else gain(old, nn["marker_s"])
        )

    w("## Speed\n")
    w(
        "The new path's time is the first run, including the store open (cold); the warm time (median of repeats, "
        "no open) is in parentheses. Gains are given as cold–warm.\n"
    )
    w("| Typical use | Old path | nmzarr + app kernel | Gain |")
    w("|---|---|---|---|")
    for n in cohorts:
        who = "one subject" if n == 1 else f"{n} subjects"
        nn = cn[f"zscores/{n}"]
        new_t = f"**{dur(nn['marker_s'])}** ({dur(nn['marker_s_warm'])} warm)"
        z14 = c14[f"zscores/{n}"]["marker_s"]
        w(
            f"| **Score {who}**, whole marker, like for like: 1.4.0 fork, same models, `compute_zscores` | "
            f"{dur(z14)} (scaled) | {new_t} | **{gain_range(z14, nn)}** |"
        )
        for mode, what in (
            ("predict", "`predict` (what proc_nm n01 calls)"),
            ("zscores", "`compute_zscores` only"),
        ):
            for label, c in (("1.3.0", c13), ("1.4.0 fork", c14)):
                if label == "1.4.0 fork" and mode == "zscores":
                    continue
                a = c[f"{mode}/{n}"]["marker_s"]
                w(
                    f"| Score {who}, whole marker, {label} {what} | {dur(a)} (scaled) | {new_t} | {gain_range(a, nn)} |"
                )
    nn1 = cn["zscores/1"]
    read13 = (
        o13["unzip_s_all_archives"]
        + n_models * c13["zscores/1"]["ms_per_voxel"]["load"] / 1e3
    )
    read14 = (
        o14["unzip_s_all_archives"]
        + n_models * c14["zscores/1"]["ms_per_voxel"]["load"] / 1e3
    )
    w(
        f"| **Read** one marker's models (unpack + `NormativeModel.load` vs store read) | 1.3.0 {dur(read13)}; "
        f"1.4.0 fork {dur(read14)} | {dur(new['load_ms_first'] / 1e3)} first ({dur(new['load_ms'] / 1e3)} warm) | "
        f"{gain(read13, new['load_ms_first'] / 1e3)} vs 1.3.0; {gain(read14, new['load_ms_first'] / 1e3)} vs fork |"
    )
    p13, p14 = c13["predict/1"]["ms_per_voxel"], c14["predict/1"]["ms_per_voxel"]
    z13, z14 = c13["zscores/1"]["ms_per_voxel"], c14["zscores/1"]["ms_per_voxel"]
    us = nn1["us_per_voxel_per_subject"]
    new_all = new["us_per_voxel_read"] + us
    w(
        f"| **Per voxel**, one subject: read; build inputs; z only; `predict` | "
        f"1.3.0: {o13['ms_per_voxel_read'] + p13['load']:.2f} ms; {p13['normdata']:.1f} ms; {z13['call']:.1f} ms; {p13['call']:.1f} ms. "
        f"Fork: {o14['ms_per_voxel_read'] + p14['load']:.2f} ms; {p14['normdata']:.1f} ms; {z14['call']:.1f} ms; {p14['call']:.1f} ms | "
        f"read {new['us_per_voxel_read']:.1f} µs; none; scoring {us:.2f} µs; — | "
        f"z with read, inputs and scoring: {gain(z13['total'], new_all / 1e3)} vs 1.3.0, {gain(z14['total'], new_all / 1e3)} vs fork; "
        f"scoring alone: {gain(z14['call'], us / 1e3)} vs fork |"
    )
    nb = cn[f"zscores/{big}"]
    w(
        f"| **Per subject** in a {big}-subject cohort, whole marker | "
        f"1.3.0 {dur(c13[f'predict/{big}']['marker_s'] / big)}; fork {dur(c14[f'predict/{big}']['marker_s'] / big)} | "
        f"{dur(nb['score_ms'] / 1e3 / big)} + read once | |"
    )
    m14 = c14["predict/1"]["ms_per_voxel_total_median_q1_q3"]
    w("")
    w(
        f"Spread of the old path: the fork's `predict` per voxel has median {m14[0]:.1f} ms, quartiles {m14[1]:.1f}–{m14[2]:.1f} ms "
        f"over {o14['voxels_timed']} voxels (one pass each).\n"
    )

    w("### Inside one `predict` call (ms per voxel, one subject)\n")
    w("| Step | PCNtoolkit 1.3.0 | PCNtoolkit 1.4.0 fork | Needed for a z-score? |")
    w("|---|---|---|---|")
    need = {
        "compute_zscores": "yes",
        "compute_centiles": "no",
        "compute_yhat": "no",
        "compute_baseline_logp": "no",
        "compute_logp": "no",
    }
    s13, s14 = (
        c13["predict/1"]["ms_per_voxel_by_step"],
        c14["predict/1"]["ms_per_voxel_by_step"],
    )
    for k, needed in need.items():
        w(f"| `{k}` | {s13.get(k, 0):.1f} | {s14.get(k, 0):.1f} | {needed} |")
    w(
        f"| other (`predict` bookkeeping) | {p13['call'] - sum(s13.values()):.1f} | {p14['call'] - sum(s14.values()):.1f} | |"
    )
    w(
        f"| `NormData.from_dataframe` (before the call) | {p13['normdata']:.1f} | {p14['normdata']:.1f} | no |"
    )
    w("")

    w("## Storage (one marker)\n")
    a13, a14 = st["archives"]["pcn13"], st["archives"]["pcn14"]
    sm = st["store"]["markers"][man["marker"]]
    shared = sum(a["bytes"] for a in st["store"]["shared"].values())
    w(
        "| | Archives, PCNtoolkit 1.3.0 | Archives, PCNtoolkit 1.4.0 fork | nmzarr store | Gain vs 1.4.0 archives |"
    )
    w("|---|---|---|---|---|")
    w(
        f"| **On disk** | {size(a13['bytes'])} | {size(a14['bytes'])} | **{size(sm['bytes'])}** (+ {size(shared)} grid, shared by all markers) | "
        f"**{gain(a14['bytes'], sm['bytes'])}** |"
    )
    w(
        f"| Bytes per modelled voxel | {a13['bytes_per_model']:,.0f} | {a14['bytes_per_model']:,.0f} | {sm['bytes'] / n_models:,.0f} | "
        f"{gain(a14['bytes_per_model'], sm['bytes'] / n_models)} |"
    )
    w(
        f"| Files or objects | {a13['files']} zips holding {a13['json_entries']:,} JSON files | {a14['files']} zips holding "
        f"{a14['json_entries']:,} JSON files | {sum(a['objects'] for a in sm['arrays'].values())} objects "
        f"({len(sm['arrays'])} arrays) | |"
    )
    w(
        f"| **What the app downloads** | whole archives | whole archives | **{size(sm['predict_bytes'])}** in "
        f"{sm['predict_objects']} objects (9 predict arrays) | {gain(a14['bytes'], sm['predict_bytes'])} |"
    )
    w(
        f"| In memory, decoded | | | {size(sm['predict_decoded_bytes'])} (typed arrays) | |"
    )
    j13, j14 = a13["voxel_json"], a14["voxel_json"]
    w(
        f"| One voxel's model | {j13['json_bytes']:,} B of JSON, D = {j13['D']} | {j14['json_bytes']:,} B of JSON, D = {j14['D']}; "
        f"{j14['json_bytes_predict_fields']:,} B of it are the numbers prediction uses | "
        f"{j14['float64_values_predict']} float64 = {j14['binary_bytes_predict']} B before compression | "
        f"{gain(j14['json_bytes'], j14['binary_bytes_predict'])} |"
    )
    w("")
    w("### Store arrays\n")
    w("| Array | Shape | dtype | Decoded | On disk | Ratio | Read by the app |")
    w("|---|---|---|---|---|---|---|")
    predict = {
        "has_model",
        "m",
        "A_triu",
        "hyp",
        "mean_knots",
        "inscaler",
        "outscaler",
        "be_map",
        "cov_ranges",
    }
    for name, a in sorted(sm["arrays"].items(), key=lambda kv: -kv[1]["bytes"]):
        w(
            f"| `{name}` | {' × '.join(map(str, a['shape']))} | {a['dtype']} | {size(a['decoded_bytes'])} | {size(a['bytes'])} | "
            f"{a['decoded_bytes'] / max(a['bytes'], 1):.1f} | {'yes' if name in predict else 'no'} |"
        )
    w("")
    (r / "summary.md").write_text("\n".join(L) + "\n")
    print(f"wrote {r / 'summary.md'}")


if __name__ == "__main__":
    main()
