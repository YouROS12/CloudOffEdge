"""Paper-2 figures, one consistent visual system, computed from raw results.

  python experiments/scripts/make_paper_figures.py            # -> paper/manuscript/figures/*.pdf|png

Inputs (offline, all on disk):
  outputs/edge_v3l_predictions_test.csv   per-image edge prediction + margin
  outputs/cloud_predictions_test.csv      per-image cloud prediction
  results/option_A_3arch/*.csv            sweeps, arch benchmark
  results/option_B_verifier_disagree/     verifier sweep

Style: reference palette of the data-viz method (slots 1-3 blue/orange/aqua,
validated all-pairs), hairline recessive grid, thin marks, selective direct
labels, status colours only where a mark means correct/wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(PAPER2 / "experiments" / "client"))
from fleet_sim import THEMED_FARMS, crop_of  # noqa: E402

# ---------------------------------------------------------------- palette
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, BLUE_DARK = "#b7d3f6", "#184f95"
NEUTRAL = "#b5b3ab"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
CLOUD_ACC, TARGET = 0.8017, 0.7617
MSG_KB = 2 * (1280 + 89) / 1024  # verifier's always-on message (fp16 embedding + softmax)
# S in Eq. (3): mean escalation payload measured in the baseline deployment (448 px, q90), so offline
# bandwidth is in the same units as the deployment. Sweeps store rates; KB = rate x S.
_SUMMARY = PAPER2 / "results" / "fleet_run" / "summary.json"
IMAGE_KB = (__import__("json").loads(_SUMMARY.read_text(encoding="utf-8"))["overall"]["cloud_only_kb_per_image"]
            if _SUMMARY.exists() else 250.0)
TAU = 0.138424  # validation-calibrated, results/threshold_calibration.json

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0,
    "ytick.major.size": 0, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300,
    "pdf.fonttype": 42, "lines.solid_capstyle": "round", "axes.axisbelow": True,
})


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def load_per_image(outputs: Path) -> pd.DataFrame:
    e = pd.read_csv(outputs / "edge_v3l_predictions_test.csv")
    c = pd.read_csv(outputs / "cloud_predictions_test.csv")
    for d in (e, c):
        d["key"] = d["true_label"] + "/" + d["image_path"].str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    df = e.merge(c[["key", "cloud_pred"]], on="key", validate="one_to_one")
    crops = [c for _, cs in THEMED_FARMS.values() for c in cs]
    crop_farm = {c: f for f, (_, cs) in THEMED_FARMS.items() for c in cs}
    df["crop"] = df["true_label"].map(lambda s: crop_of(s, crops))
    df["crop"] = df["crop"].replace({"grapevine": "grape"})
    df["farm"] = df["crop"].map(lambda c: crop_farm.get(c, crop_farm.get("grape")))
    df["edge_ok"] = df["edge_pred"] == df["true_label"]
    df["cloud_ok"] = df["cloud_pred"] == df["true_label"]
    df["esc"] = df["edge_margin"] < TAU
    df["final_ok"] = np.where(df["esc"], df["cloud_ok"], df["edge_ok"])
    return df


# ---------------------------------------------------------------- 1. flow
def band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.55):
    """Smooth band from [y0a,y0b] at x0 to [y1a,y1b] at x1."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b),
             (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig_flow(df: pd.DataFrame, out: Path) -> None:
    n = len(df)
    groups = {  # (edge_ok, escalated, final_ok) -> count
        k: int(((df.edge_ok == k[0]) & (df.esc == k[1]) & (df.final_ok == k[2])).sum())
        for k in [(a, b, c) for a in (True, False) for b in (False, True) for c in (True, False)]}
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_axis_off()
    W, gap = 0.035, 0.03          # column width, gap between blocks (fraction of height)
    xs = [0.06, 0.47, 0.88]

    def column(x, sizes, labels, colors):
        tops, y = {}, 1.0
        total_gap = gap * (len(sizes) - 1)
        for key, s in sizes.items():
            h = s / n * (1 - total_gap)
            ax.add_patch(FancyBboxPatch((x - W / 2, y - h), W, h, boxstyle="round,pad=0,rounding_size=0.006",
                                        facecolor=colors[key], edgecolor="none"))
            tops[key] = [y, y - h]
            ax.text(x + (W / 2 + 0.012) * (1 if x < 0.9 else -1), y - h / 2,
                    f"{labels[key]}\n{s:,} ({s / n:.1%})", ha="left" if x < 0.9 else "right",
                    va="center", fontsize=7.5, color=INK, linespacing=1.25)
            y -= h + gap
        return tops

    e_sizes = {True: int(df.edge_ok.sum()), False: int((~df.edge_ok).sum())}
    # order chosen so no band crosses another: local-correct, escalated, local-wrong
    r_sizes = {"local_ok": groups[(True, False, True)], "esc": int(df.esc.sum()),
               "local_bad": groups[(False, False, False)]}
    f_sizes = {True: int(df.final_ok.sum()), False: int((~df.final_ok).sum())}
    c0 = column(xs[0], e_sizes, {True: "edge correct", False: "edge wrong"}, {True: GOOD, False: CRITICAL})
    c1 = column(xs[1], r_sizes, {"local_ok": "kept local, correct", "local_bad": "kept local, wrong",
                                 "esc": f"escalated ($m<{TAU:.3f}$)"},
                {"local_ok": NEUTRAL, "local_bad": NEUTRAL, "esc": BLUE})
    c2 = column(xs[2], f_sizes, {True: "final correct", False: "final wrong"}, {True: GOOD, False: CRITICAL})

    h = lambda s: s / n * (1 - gap * 2)  # noqa: E731  (3-block columns dominate spacing)
    # cursors that walk down each block as flows leave / arrive
    out0 = {k: v[0] for k, v in c0.items()}
    in1 = {k: v[0] for k, v in c1.items()}
    out1 = {k: v[0] for k, v in c1.items()}
    in2 = {k: v[0] for k, v in c2.items()}

    def flow(src_cur, sk, dst_cur, dk, count, color):
        if count == 0:
            return
        hh = count / n * (1 - gap * (len(c1) - 1))
        band(ax, xs_src[0] + W / 2, src_cur[sk], src_cur[sk] - hh, xs_src[1] - W / 2,
             dst_cur[dk], dst_cur[dk] - hh, color)
        src_cur[sk] -= hh
        dst_cur[dk] -= hh

    xs_src = (xs[0], xs[1])
    flow(out0, True, in1, "local_ok", groups[(True, False, True)], NEUTRAL)
    flow(out0, True, in1, "esc", groups[(True, True, True)] + groups[(True, True, False)], BLUE)
    flow(out0, False, in1, "esc", groups[(False, True, True)] + groups[(False, True, False)], BLUE)
    flow(out0, False, in1, "local_bad", groups[(False, False, False)], NEUTRAL)
    xs_src = (xs[1], xs[2])
    flow(out1, "local_ok", in2, True, groups[(True, False, True)], NEUTRAL)
    rescued = groups[(False, True, True)]
    harmed = groups[(True, True, False)]
    flow(out1, "esc", in2, True, groups[(True, True, True)] + rescued, BLUE)
    flow(out1, "esc", in2, False, groups[(False, True, False)] + harmed, BLUE)
    flow(out1, "local_bad", in2, False, groups[(False, False, False)], NEUTRAL)

    for x, t in zip(xs, ["Edge model (MobileNetV3-L)", "Routing decision", "Delivered diagnosis"]):
        ax.text(x, 1.075, t, ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="semibold")
    ax.text(0.675, -0.10, f"escalation rescued {rescued:,} edge errors; harmed {harmed:,} correct edge "
            f"predictions (net +{rescued - harmed:,})", ha="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.14, 1.12)
    save(fig, out, "fig_flow")
    print(f"   tau={TAU}: escalation {df.esc.mean():.1%}, final acc {df.final_ok.mean():.2%}, "
          f"edge acc {df.edge_ok.mean():.2%}, rescued {rescued}, harmed {harmed}")


# ---------------------------------------------------------------- 2. pareto
def fig_pareto(results: Path, out: Path) -> None:
    sw = pd.read_csv(results / "option_A_3arch" / "three_arch_sweep.csv")
    ver = pd.read_csv(results / "option_B_verifier_disagree" / "verifier_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # criterion region
    ax.fill_between([0, 0.30 * IMAGE_KB], TARGET * 100, 88, color=BLUE_LIGHT, alpha=0.18, lw=0)
    ax.text(3 * IMAGE_KB / 250.0, 87.6, "criterion: $\\geq$95 % of cloud accuracy at $\\leq$30 % escalation",
            fontsize=7, color=BLUE_DARK, va="top")
    k = IMAGE_KB / 250.0  # layout positions were designed for a 250 KB axis
    ax.axhline(CLOUD_ACC * 100, color=MUTED, lw=0.8)
    ax.text(247 * k, CLOUD_ACC * 100 + 0.25, "always-cloud 80.2 %", ha="right", va="bottom", fontsize=7, color=INK2)
    ax.axhline(TARGET * 100, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(247 * k, TARGET * 100 - 0.25, "95 % target", ha="right", va="top", fontsize=7, color=INK2)

    def curve(arch, signal):
        d = sw[(sw.arch == arch) & (sw.signal == signal)].sort_values("offload_rate")
        return d.offload_rate.values * IMAGE_KB, d.accuracy.values * 100

    x, y = curve("v3-Large", "oracle")
    ax.plot(x, y, color=BLUE_LIGHT, lw=1.2, label="oracle router, MobileNetV3-L (upper bound)")
    x, y = curve("v3-Small", "margin")
    ax.plot(x, y, color=NEUTRAL, lw=1.6, label="margin, MobileNetV3-S edge")
    v = ver[ver.signal == "verifier"].sort_values("offload_rate")
    ax.plot(MSG_KB + v.offload_rate * IMAGE_KB, v.accuracy * 100, color=ORANGE, lw=1.6,
            label="cloud-supervised verifier")
    x, y = curve("v3-Large", "margin")
    ax.plot(x, y, color=BLUE, lw=2.0, label="margin, MobileNetV3-L edge")

    op_rate, op_acc = 0.18602, 75.82  # results/threshold_calibration.json, test split
    op_kb = op_rate * IMAGE_KB
    ax.scatter([0], [69.32], s=36, color=BLUE, zorder=5, edgecolor=SURFACE, linewidth=1.5)
    ax.text(6 * k, 68.0, "edge only\n69.3 %", fontsize=7, color=INK2, va="center", linespacing=1.2)
    ax.scatter([op_kb], [op_acc], s=70, facecolor=SURFACE, edgecolor=BLUE_DARK, linewidth=1.8, zorder=6)
    ax.annotate(f"operating point, $\\tau$ calibrated on validation\n{op_kb:.1f} KB · 18.6 % escalated · 75.8 %\n"
                "5.4$\\times$ less uplink than always-cloud",
                (op_kb, op_acc), (118 * k, 71.0), fontsize=7.2, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set_xlim(-4 * k, 252 * k)
    ax.set_ylim(60, 88)
    ax.set_xlabel("mean uplink per diagnosis (KB)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.legend(loc="lower right", handlelength=1.6)
    save(fig, out, "fig_pareto")


# ---------------------------------------------------------------- 3. capacity
def fig_capacity(results: Path, out: Path) -> None:
    sw = pd.read_csv(results / "option_A_3arch" / "three_arch_sweep.csv")
    bench = pd.read_csv(results / "option_A_3arch" / "arch_benchmark.csv").set_index("arch")
    archs = [("v3-Small", "MobileNetV3-S", NEUTRAL), ("v3-Large", "MobileNetV3-L", BLUE),
             ("EfficientNet-B0", "EfficientNet-B0", ORANGE)]
    # exact escalation @95 % (image by image) and edge accuracy: experiments/scripts/review_analyses.py
    ex = json.loads((results / "review" / "analyses.json").read_text(encoding="utf-8"))["exact_at_95"]
    need = {k: 100 * ex[k]["esc_for_95"] for k in ("v3-Small", "v3-Large", "EfficientNet-B0")}
    edge_acc = {k: 100 * ex[k]["edge_acc"] for k in ("v3-Small", "v3-Large", "EfficientNet-B0")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.3))

    a.axhline(TARGET * 100, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    a.axvline(30, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    a.text(30.8, 61.2, "30 % budget", fontsize=7, color=INK2)
    a.text(99, TARGET * 100 + 0.3, "95 % target", fontsize=7, color=INK2, ha="right", va="bottom")
    for arch, label, col in archs:
        d = sw[(sw.arch == arch) & (sw.signal == "margin")].sort_values("offload_rate")
        a.plot(d.offload_rate * 100, d.accuracy * 100, color=col, lw=1.8 if arch == "v3-Large" else 1.4,
               label=label, zorder=3 if arch == "v3-Large" else 2)
        a.scatter([need[arch]], [TARGET * 100], s=26, color=col, edgecolor=SURFACE, lw=1.2, zorder=4)
    a.annotate(f"{need['v3-Small']:.1f} %", (need["v3-Small"], TARGET * 100), (49, 72.5), fontsize=7,
               color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.annotate(f"{need['v3-Large']:.1f} %", (need["v3-Large"], TARGET * 100), (8, 78.6), fontsize=7,
               color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    a.set_xlim(0, 100)
    a.set_ylim(60, 81.5)
    a.set_xlabel("escalation rate (%)")
    a.set_ylabel("top-1 accuracy (%)")
    a.set_title("(a) accuracy vs. escalation, margin router")
    a.legend(loc="lower right")

    b.axhline(30, color=MUTED, lw=0.8, ls=(0, (1, 2)))
    for arch, label, col in archs:
        gm = bench.loc[arch, "flops_G_macs_b1"]
        pm = bench.loc[arch, "params_M"]
        ok = need[arch] <= 30
        b.scatter([gm], [need[arch]], s=pm * 95, color=col, alpha=0.85, edgecolor=SURFACE, lw=2, zorder=3)
        place = {"v3-Small": (gm + 0.035, need[arch], "left"),
                 "v3-Large": (gm, need[arch] + 6.5, "center"),
                 "EfficientNet-B0": (gm, need[arch] - 8.5, "center")}[arch]
        b.text(place[0], place[1], f"{label} — {'meets' if ok else 'misses'} target\n"
               f"{pm:.2f} M params · edge {edge_acc[arch]:.2f} %",
               ha=place[2], va="center", fontsize=6.8, color=INK, linespacing=1.25)
    b.text(0.465, 30.8, "30 % budget", fontsize=7, color=INK2, ha="right", va="bottom")
    b.set_xlim(0, 0.47)
    b.set_ylim(5, 55)
    b.set_xlabel("edge compute (GMACs per image)")
    b.set_ylabel("escalation needed for 95 % target (%)")
    b.set_title("(b) cost of the edge vs. escalation it saves")
    save(fig, out, "fig_capacity")


# ---------------------------------------------------------------- 4. bandwidth tax
def fig_tax(results: Path, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1.45], wspace=0.55))
    A = json.loads((results / "review" / "analyses.json").read_text(encoding="utf-8"))
    s_ = A["calibration"]["v3-Large"]["signals"]
    best_v = max(A["verifiers"].values(), key=lambda v: v["test_auc_useful"])
    sig = [("1 − margin", s_["margin"]["test_auc_useful"], BLUE),
           ("1 − max. probability", s_["max_prob"]["test_auc_useful"], NEUTRAL),
           ("entropy", s_["entropy"]["test_auc_useful"], NEUTRAL),
           ("cloud-supervised verifier", best_v["test_auc_useful"], ORANGE)]
    for i, (name, auc, col) in enumerate(sig):
        a.plot([0.5, auc], [i, i], color=GRID, lw=1.5, zorder=1)
        a.scatter([auc], [i], s=40, color=col, edgecolor=SURFACE, lw=1.5, zorder=3)
        a.text(auc + 0.008, i, f"{auc:.3f}", va="center", fontsize=7.2, color=INK)
    a.set_yticks(range(len(sig)), [s[0] for s in sig])
    a.invert_yaxis()
    a.set_xlim(0.5, 0.86)
    a.set_ylim(len(sig) - 0.5, -0.6)
    a.grid(axis="y", visible=False)
    a.set_xlabel("AUC for useful escalations")
    a.set_title("(a) ranking quality")

    # escalation rate at which each router's test curve first reaches 95 % of cloud accuracy
    r_m = A["exact_at_95"]["v3-Large"]["esc_for_95"]
    r_v = best_v["esc_for_95"]
    rows = [("margin", 0.0, r_m * IMAGE_KB), ("verifier", MSG_KB, r_v * IMAGE_KB), ("always-cloud", 0.0, IMAGE_KB)]
    gap = IMAGE_KB / 250.0
    for i, (name, msg, img) in enumerate(rows):
        if msg:
            b.barh(i, msg, color=ORANGE, height=0.5)
        b.barh(i, img, left=msg + (gap if msg else 0), color=BLUE if name != "always-cloud" else NEUTRAL,
               height=0.5)
        b.text(msg + img + 4 * gap, i, f"{msg + img:.1f} KB", va="center", fontsize=7.2, color=INK)
    b.text(0, 1.33, f"↑ always-on router message: {MSG_KB:.2f} KB for every image", fontsize=6.6,
           color=ORANGE, va="top")
    b.set_yticks(range(3), [f"margin rule\n({r_m:.1%} escalated)".replace("%", " %"),
                            f"verifier\n({r_v:.1%} escalated)".replace("%", " %"), "always-cloud\n(100 %)"])
    b.invert_yaxis()
    b.set_xlim(0, IMAGE_KB * 1.16)
    b.grid(axis="y", visible=False)
    b.set_xlabel("uplink per diagnosis at 95 % of cloud accuracy (KB)")
    b.set_title("(b) the bandwidth tax of a learned router")
    save(fig, out, "fig_tax")


# ---------------------------------------------------------------- 5. per-crop
def fig_crops(df: pd.DataFrame, out: Path) -> None:
    g = df.groupby("crop").agg(n=("edge_ok", "size"), edge=("edge_ok", "mean"), final=("final_ok", "mean"),
                               esc=("esc", "mean"), farm=("farm", "first"))
    g = g[g.n >= 20].sort_values("esc")
    farm_name = {f: t for f, (t, _) in THEMED_FARMS.items()}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 0.19 * len(g) + 0.9), sharey=True,
                               gridspec_kw=dict(width_ratios=[2.3, 1], wspace=0.06))
    y = np.arange(len(g))
    for yi, (_, r) in zip(y, g.iterrows()):
        col = BLUE if r.final >= r.edge else ORANGE
        a.plot([r.edge * 100, r.final * 100], [yi, yi], color=col, lw=1.6, alpha=0.6, zorder=1)
    a.scatter(g.edge * 100, y, s=16, color=NEUTRAL, zorder=2, label="edge only", edgecolor=SURFACE, lw=0.8)
    a.scatter(g.final * 100, y, s=20, color=BLUE, zorder=3, label=f"with escalation ($\\tau$={TAU:.3f})",
              edgecolor=SURFACE, lw=0.8)
    a.set_yticks(y, [f"{c} (n={g.loc[c, 'n']})  ·  {g.loc[c, 'farm']}" for c in g.index])
    a.set_xlabel("top-1 accuracy (%)")
    a.set_xlim(30, 101)
    a.set_title("accuracy per crop: edge only → delivered")
    a.legend(loc="lower left", handletextpad=0.3)
    a.grid(axis="y", visible=False)
    b.axvline(df.esc.mean() * 100, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    b.barh(y, g.esc * 100, color=BLUE_LIGHT, height=0.62, zorder=2)
    for yi, v in zip(y, g.esc):
        b.text(v * 100 + 1, yi, f"{v:.0%}", va="center", fontsize=6.5, color=INK2, zorder=3,
               bbox=dict(boxstyle="square,pad=0.08", facecolor=SURFACE, edgecolor="none"))
    b.set_xlim(0, max(60, g.esc.max() * 100 + 10))
    b.set_xlabel("escalated (%)")
    b.set_title("escalation rate")
    b.grid(axis="y", visible=False)
    a.set_ylim(-0.7, len(g) - 0.3)
    fig.text(0.01, 0.005, "Farm letters: " + ", ".join(f"{f} = {farm_name[f]}" for f in "ABCD")
             + f".  Dashed line: overall escalation rate.  Crops with ≥ 20 test images.",
             fontsize=6.8, color=MUTED)
    save(fig, out, "fig_crops")


# ---------------------------------------------------------------- 6-8. fleet run (deployment)
FARM_COLORS = {"A": BLUE, "B": ORANGE, "C": AQUA, "D": "#eda100"}


def load_fleet(run: Path) -> pd.DataFrame:
    ev = pd.read_csv(run / "events.csv")
    ev["h"] = ev.sim_day * 24 + ev.sim_hour
    return ev


def fig_fleet_timeline(ev: pd.DataFrame, days: int, out: Path, ev2: pd.DataFrame | None = None) -> None:
    farms = sorted(ev.farm.unique())
    H = days * 24
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 3.9), sharex=True,
                               gridspec_kw=dict(height_ratios=[1, 1.15], hspace=0.12))
    for ax in (a, b):
        for d in range(days):
            ax.axvspan(d * 24 - 5, d * 24 + 7, color="#f3f2ee", lw=0, zorder=0)
    bottom = np.zeros(H)
    esc = ev[ev.route_decision == 1]
    for f in farms:
        counts = np.bincount(esc[esc.farm == f].h, minlength=H)[:H]
        a.bar(np.arange(H), counts, bottom=bottom, width=0.85, color=FARM_COLORS.get(f, NEUTRAL),
              label=f"farm {f}", linewidth=0)
        bottom += counts
    a.set_ylabel("escalations\nper sim. hour")
    a.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=4))
    a.legend(ncol=len(farms), loc="upper left", bbox_to_anchor=(0, 1.22), handlelength=1, columnspacing=1)
    a.grid(axis="x", visible=False)
    up = np.cumsum(np.bincount(ev.h, weights=ev.bandwidth_kb, minlength=H)[:H]) / 1024
    base = np.cumsum(np.bincount(ev.h, weights=ev.cloud_only_kb, minlength=H)[:H]) / 1024
    b.plot(np.arange(H), base, color=NEUTRAL, lw=1.8, label="always-cloud (same payload)")
    b.plot(np.arange(H), up, color=BLUE, lw=2.0,
           label="selective escalation" + (", original edge" if ev2 is not None else ""))
    b.fill_between(np.arange(H), up, base, color=BLUE_LIGHT, alpha=0.35, lw=0)
    b.text(H - 1, base[-1], f"  {base[-1]:.1f} MB", va="center", fontsize=7.2, color=INK2)
    b.text(H - 1, up[-1], f"  {up[-1]:.1f} MB", va="center", fontsize=7.2, color=BLUE_DARK)
    b.text(H * 0.52, (up[-1] + base[-1]) / 2.2, f"{base[-1] / max(up[-1], 1e-9):.1f}$\\times$ less uplink",
           fontsize=8, color=BLUE_DARK, ha="center")
    if ev2 is not None:  # distilled deployment: distilled edge model, same images and payload
        up2 = np.cumsum(np.bincount(ev2.h, weights=ev2.bandwidth_kb, minlength=H)[:H]) / 1024
        b.plot(np.arange(H), up2, color=AQUA, lw=2.0, label="selective escalation, distilled edge")
        b.text(H - 1, up2[-1], f"  {up2[-1]:.1f} MB", va="top", fontsize=7.2, color=INK)
    b.set_ylabel("cumulative uplink (MB)")
    b.set_xticks([d * 24 + 12 for d in range(days)], [f"day {d + 1}" for d in range(days)])
    b.set_xlim(-1, H + 10)
    b.legend(loc="upper left")
    b.grid(axis="x", visible=False)
    save(fig, out, "fig_fleet_timeline")


def fig_fleet_latency(ev: pd.DataFrame, out: Path) -> None:
    ok = ev[(ev.route_decision == 0) | (ev.cloud_ok == 1)]
    loc, esc = ok[ok.route_decision == 0], ok[(ok.route_decision == 1) & (ok.cloud_ok == 1)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.35))
    rng = np.random.default_rng(0)
    for i, (d, col, lab) in enumerate([(loc, NEUTRAL, "local"), (esc, BLUE, "escalated")]):
        if len(d) == 0:
            continue
        y = i + rng.uniform(-0.28, 0.28, len(d))
        a.scatter(d.latency_ms / 1000, y, s=5, color=col, alpha=0.35, lw=0)
        med = d.latency_ms.median() / 1000
        a.plot([med, med], [i - 0.36, i + 0.36], color=INK, lw=1.6)
        a.text(med, i + 0.42, f"median {med:.2f} s" if med >= 1 else f"median {med * 1000:.0f} ms",
               ha="center", fontsize=7, color=INK)
    a.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    a.set_xticks(ticks, ["50 ms", "100 ms", "200 ms", "500 ms", "1 s", "2 s"])
    a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    a.set_xlim(0.04, 2.5)
    a.set_yticks([0, 1], [f"local\n(n={len(loc):,})", f"escalated\n(n={len(esc):,})"])
    a.set_ylim(-0.6, 1.75)
    a.set_xlabel("time until the farmer gets an answer (s, log scale)")
    a.set_title("(a) answer on the farm, every image")
    a.grid(axis="y", visible=False)
    parts = [("edge inference, Pi 5", esc.edge_ms.median(), NEUTRAL),
             ("upload to cloud queue", esc.network_ms.median(), "#eda100"),
             ("ensemble compute, Nano", esc.cloud_ms.median(), BLUE)]
    for i, (name, v, col) in enumerate(parts):
        if not np.isfinite(v):
            continue
        b.barh(i, v / 1000, color=col, height=0.55)
        txt = f"{v:.0f} ms" if v < 1000 else f"{v / 1000:.2f} s"
        b.text(v / 1000 + 0.12, i, txt, va="center", fontsize=7.2, color=INK)
    b.set_yticks(range(len(parts)), [p[0] for p in parts])
    b.invert_yaxis()
    b.set_xlim(0, max(p[1] for p in parts if np.isfinite(p[1])) / 1000 * 1.3)
    b.set_xlabel("median per escalated image (s)")
    b.set_title("(b) cost of an escalation, by stage")
    b.grid(axis="y", visible=False)
    save(fig, out, "fig_fleet_latency")


def fig_fleet_vs_offline(ev: pd.DataFrame, df: pd.DataFrame, out: Path) -> None:
    """Per farm: escalation rate predicted from offline margins vs measured on device."""
    farms = sorted(ev.farm.unique())
    offline = df.groupby("farm").esc.mean()
    measured = ev.groupby("farm").route_decision.mean()
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for i, f in enumerate(farms):
        o, m = offline.get(f, np.nan) * 100, measured.get(f, np.nan) * 100
        ax.plot([o, m], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([o], [i], s=26, color=NEUTRAL, zorder=2, edgecolor=SURFACE, lw=1)
        ax.scatter([m], [i], s=30, color=FARM_COLORS.get(f, BLUE), zorder=3, edgecolor=SURFACE, lw=1)
        ax.text(max(o, m) + 0.8, i, f"{m:.1f} %", va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(farms)), [f"farm {f} · {THEMED_FARMS[f][0]}" for f in farms])
    ax.invert_yaxis()
    ax.set_xlabel("escalation rate (%)")
    ax.set_title("offline prediction (grey) vs on device")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_fleet_vs_offline")


# ---------------------------------------------------------------- Nano port
BACKBONE_STYLE = {"dinov2_vitl14": ("DINOv2-L", BLUE), "dinov3_vitl16": ("DINOv3-L", ORANGE),
                  "clip_vitl14": ("CLIP-L", AQUA)}


def load_tegrastats(path: Path) -> pd.DataFrame:
    import re
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+\.\d+) RAM (\d+)/(\d+)MB", line)
        p = re.search(r"POM_5V_IN (\d+)/", line)
        if not (m and p):
            continue
        g = re.search(r"GR3D_FREQ (\d+)%", line)
        rows.append({"t": float(m.group(1)), "ram_mb": int(m.group(2)), "ram_total_mb": int(m.group(3)),
                     "p_in_w": int(p.group(1)) / 1000, "gpu_pct": int(g.group(1)) if g else np.nan})
    return pd.DataFrame(rows)


def flush_profile(prof_dir: Path) -> dict:
    """Per run: energy, mean power, peak RAM, backbone phase spans (from the flush timings)."""
    runs = json.loads((prof_dir / "runs.json").read_text(encoding="utf-8"))["runs"]
    ts = load_tegrastats(prof_dir / "tegrastats.log")
    outs = []
    for r in runs:
        idle = ts[(ts.t >= r["t_idle"] + 5) & (ts.t < r["t_enqueue"])]
        fl = ts[(ts.t >= r["t_flush_start"]) & (ts.t <= r["t_flush_end"])]
        dur = r["t_flush_end"] - r["t_flush_start"]
        e_total = np.trapezoid(fl.p_in_w, fl.t)
        p_idle = idle.p_in_w.mean()
        spans, t = [], r["t_flush_start"]
        for name in BACKBONE_STYLE:
            d = (r["load_ms"][name] + sum(tm[name] for tm in r["timings_ms"])) / 1000
            spans.append((name, t - r["t_flush_start"], t + d - r["t_flush_start"], r["load_ms"][name] / 1000))
            t += d
        outs.append({"n": r["n"], "dur_s": dur, "e_total_j": e_total, "p_idle_w": p_idle,
                     "p_mean_w": fl.p_in_w.mean(), "p_max_w": fl.p_in_w.max(),
                     "e_per_img_j": e_total / r["n"], "e_incr_per_img_j": (e_total - p_idle * dur) / r["n"],
                     "ram_idle_mb": idle.ram_mb.median(), "ram_peak_mb": fl.ram_mb.max(),
                     "ram_total_mb": int(ts.ram_total_mb.iloc[0]), "spans": spans,
                     "load_s": sum(r["load_ms"].values()) / 1000, "t0": r["t_idle"], "trace": ts[
                         (ts.t >= r["t_idle"]) & (ts.t <= r["t_end"])].assign(x=lambda d: d.t - r["t_flush_start"])})
    return {"runs": outs}


def fig_port(results: Path, out: Path, prof: dict | None) -> None:
    """(a) residual-stream range per block vs the fp16 limit; (b) memory steps of the port."""
    act = json.loads((results / "activation_ranges.json").read_text(encoding="utf-8"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.42))
    fp16 = act["fp16_max"]
    a.axhspan(fp16, 1e6, color=ORANGE, alpha=0.07, lw=0)
    a.axhline(fp16, color=INK2, lw=0.8, ls=(0, (4, 3)))
    a.text(0.6, fp16 / 1.35, "fp16 maximum (65 504)", fontsize=6.8, color=INK2, va="top")
    for name, (label, col) in BACKBONE_STYLE.items():
        r = act["backbones"][name]["residual_max_per_block"]
        a.plot(range(1, len(r) + 1), r, color=col, lw=1.8, marker="o", ms=2.6)
        a.text(len(r) + 0.7, r[-1], label, fontsize=7, color=INK, va="center")
    a.set_yscale("log")
    a.set_ylim(1, 1e6)
    a.set_xlim(0, 29)
    a.set_xticks([1, 6, 12, 18, 24])
    a.set_xlabel("transformer block")
    a.set_ylabel("max |activation|, residual stream")
    a.set_title(f"(a) activation range ({act['n_images']} test images)")

    ceiling = (prof["runs"][0]["ram_total_mb"] if prof else 3956) / 1024
    steps = [("whole ViT-L: ONNX parser alone", 2.75, "fails"),
             ("6-block chunk, fused kernel graph (device)", 0.969, ""),
             ("6-block chunk as 1×1 convolutions (loaded, fp16)", 0.35, ""),
             ("all 12 chunks loaded at once (fp16 engines; 9 fit)", 12 * 0.35, "fails")]
    if prof:
        rise = np.mean([(r["ram_peak_mb"] - r["ram_idle_mb"]) / 1024 for r in prof["runs"]])
        steps.append(("chunk-major runtime: measured rise above idle", rise, ""))
    for i, (lab, v, flag) in enumerate(steps):
        col = NEUTRAL if flag else (BLUE if prof and i == len(steps) - 1 else BLUE_LIGHT)
        b.barh(i, v, color=col, height=0.42)
        b.text(0.0, i - 0.29, lab + (" — does not fit" if flag else ""), fontsize=6.9,
               color=ORANGE if flag else INK2, va="bottom")
        b.text(v + 0.06, i, f"{v:.2f} GB", va="center", fontsize=7, color=INK)
    b.axvline(ceiling, color=INK2, lw=0.8, ls=(0, (4, 3)))
    idle = (np.mean([r["ram_idle_mb"] for r in prof["runs"]]) / 1024) if prof else 1.9
    b.text(ceiling - 0.07, -1.0, f"{ceiling:.2f} GB shared by CPU and GPU\n({idle:.2f} GB in use at idle)",
           ha="right", va="center", fontsize=6.6, color=INK2, linespacing=1.25)
    b.set_yticks([])
    b.set_ylim(len(steps) - 0.5, -1.6)
    b.set_xlim(0, 5.3)
    b.grid(axis="y", visible=False)
    b.set_xlabel("memory needed (GB)")
    b.set_title("(b) fitting a ViT-L into the Nano")
    save(fig, out, "fig_port")


def fig_nano_flush(prof: dict, out: Path) -> None:
    """Memory and board power during one deployment-sized batch (run 0)."""
    r = prof["runs"][0]
    tr = r["trace"]
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 3.3), sharex=True, gridspec_kw=dict(hspace=0.12))
    for ax in (a, b):
        for name, s, e, load in r["spans"]:
            label, col = BACKBONE_STYLE[name]
            ax.axvspan(s, e, color=col, alpha=0.08, lw=0)
    for name, s, e, load in r["spans"]:
        label, col = BACKBONE_STYLE[name]
        prec = "fp16" if name == "dinov2_vitl14" else "fp32"
        a.text((s + e) / 2, r["ram_total_mb"] / 1024 * 0.985, f"{label} · {prec}", ha="center", va="top",
               fontsize=7, color=INK)
    a.plot(tr.x, tr.ram_mb / 1024, color=BLUE, lw=1.1)
    a.axhline(r["ram_total_mb"] / 1024, color=INK2, lw=0.8, ls=(0, (4, 3)))
    a.set_ylim(0, r["ram_total_mb"] / 1024 * 1.02)
    a.set_ylabel("RAM used (GB)")
    b.plot(tr.x, tr.p_in_w, color=ORANGE, lw=0.8)
    b.axhline(r["p_idle_w"], color=INK2, lw=0.8, ls=(0, (2, 2)))
    mean_idle = np.mean([q["p_idle_w"] for q in prof["runs"]])
    b.text(tr.x.max() - 2, r["p_idle_w"] - 0.15,
           f"idle {r['p_idle_w']:.1f} W in this run ({mean_idle:.1f} W mean of {len(prof['runs'])} runs)",
           fontsize=6.8, color=INK2, va="top", ha="right")
    b.set_ylabel("board power (W)")
    b.set_ylim(0, max(10, tr.p_in_w.max() * 1.1))
    b.set_xlabel(f"time since the batch started (s) — {r['n']} escalated images")
    b.set_xlim(tr.x.min(), tr.x.max())
    save(fig, out, "fig_nano_flush")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=str(Path(__file__).resolve().parents[2] / "artifacts"))
    ap.add_argument("--results", default=str(PAPER2 / "results"))
    ap.add_argument("--out", default=str(PAPER2 / "paper" / "figures"))
    ap.add_argument("--fleet-run", default="", help="fleet_sim output dir (events.csv, meta.json)")
    ap.add_argument("--only-fleet", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_per_image(Path(args.outputs))
    if not args.only_fleet:
        fig_flow(df, out)
        fig_pareto(Path(args.results), out)
        fig_capacity(Path(args.results), out)
        fig_tax(Path(args.results), out)
        fig_crops(df, out)
    prof_dir = Path(args.results) / "nano_flush_profile"
    prof = flush_profile(prof_dir) if (prof_dir / "runs.json").exists() else None
    if prof and json.loads((prof_dir / "runs.json").read_text(encoding="utf-8"))["runs"]:
        fig_nano_flush(prof, out)
    else:
        prof = None
    if (Path(args.results) / "activation_ranges.json").exists():
        fig_port(Path(args.results), out, prof)
    if args.fleet_run:
        run = Path(args.fleet_run)
        ev = load_fleet(run)
        days = int(__import__("json").loads((run / "meta.json").read_text())["args"]["days"])
        run2 = Path(args.results) / "fleet_run_improved"
        ev2 = load_fleet(run2) if (run2 / "summary.json").exists() else None
        fig_fleet_timeline(ev, days, out, ev2)
        fig_fleet_latency(ev, out)
        fig_fleet_vs_offline(ev, df, out)


if __name__ == "__main__":
    main()
