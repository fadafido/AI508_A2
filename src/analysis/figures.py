"""Figures for the report. Reads the result files, writes PNGs to results/figures/.

    python src/analysis/figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import COMPARISON, FIGURES, PAPER_REF, REPRODUCED  # noqa: E402

PAPER_C, OURS_C, THIRD_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS_LABEL = {"DS": "DS", "DSp": "DS'", "DSpp": "DS''"}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "lines.linewidth": 2,
})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200)
    plt.close(fig)
    print("wrote", FIGURES / name)


def seeds(ds, tag):
    out = []
    for f in sorted((REPRODUCED / "fgcs2025" / ds / tag).glob("seed*.json")):
        d = json.loads(f.read_text())
        out.append({"seed": int(f.stem[4:]), "AME": d["AME"], "LOOCV_HME": d["loocv"]["metrics"]["HME"],
                    "LOGOCV_HME": d["logocv"]["metrics"]["HME"]})
    return pd.DataFrame(out)


def ame_bars():
    ame = pd.read_csv(COMPARISON / "ame.csv")
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), sharey=True)
    models = ["Prats", "Dou", "Garralda"]
    x = np.arange(3)
    for ax, ds in zip(axes, ["DS", "DSp", "DSpp"]):
        sub = ame[ame.dataset == ds].set_index("model").loc[models]
        ours = sub.ours_median    # one aggregation rule everywhere: median over seeds
        ax.bar(x - 0.19, sub.paper, 0.36, color=PAPER_C, label="paper")
        ax.bar(x + 0.19, ours, 0.36, color=OURS_C, label="ours (median of seeds)")
        for i, (p, o) in enumerate(zip(sub.paper, ours)):
            if np.isfinite(o):
                ax.text(i + 0.19, o + 4, f"{o:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
            ax.text(i - 0.19, p + 4, f"{p:.0f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_xticks(x, models)
        ax.set_title(DS_LABEL[ds], fontsize=9, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("AME (s)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    save(fig, "ame_paper_vs_ours.png")


def seed_spread():
    pub = json.loads((PAPER_REF / "fgcs2025_published.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, ds, key in ((axes[0], "DSp", "table8_DSprime"), (axes[1], "DSpp", "table10_DSdoubleprime")):
        d = seeds(ds, "garralda")
        if d.empty:
            continue
        paper = pub[key]["Garralda"][5]                       # LOGOCV HME
        ax.scatter(d.seed, d.LOGOCV_HME, s=28, color=OURS_C, zorder=3, label="ours, per seed")
        ax.axhline(paper, color=PAPER_C, lw=1.5, ls="--", label=f"paper ({paper:.0f})")
        if ds == "DSpp":
            ax.set_yscale("log")
        ax.set_xticks(range(10))
        ax.set_xlabel("seed of the synthetic data")
        ax.set_title(f"Garralda model, LOGOCV HME on {DS_LABEL[ds]}", fontsize=9)
    axes[0].set_ylabel("LOGOCV HME (s)")
    axes[1].set_ylabel("LOGOCV HME (s, log scale)")
    axes[1].legend(fontsize=7.5, loc="center right")
    fig.subplots_adjust(wspace=0.35)
    save(fig, "seed_spread_dsp_dspp.png")


def sensitivity():
    rows = []
    for lam in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        tag = "garralda" if lam == 0.5 else f"garralda_lam{lam:g}"
        d = seeds("DS", tag)
        if len(d):
            rows.append(("lambda", lam, d.AME.iloc[0]))
    for k in [6, 10, 14, 18, 24, 30]:
        tag = "garralda" if k == 18 else f"garralda_kmax{k}"
        d = seeds("DS", tag)
        if len(d):
            rows.append(("kmax", k, d.AME.iloc[0]))
    df = pd.DataFrame(rows, columns=["param", "value", "AME"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
    for ax, prm, xl, default in ((axes[0], "lambda", "lambda (weight of quality vs distance)", 0.5),
                                  (axes[1], "kmax", "upper bound of k", 18)):
        s = df[df.param == prm]
        ax.plot(s.value, s.AME, "-o", color=OURS_C, ms=4)
        ax.axvline(default, color=INK2, lw=1, ls=":")
        ax.text(default, 0.03, " paper setting", fontsize=7, color=INK2, va="bottom", transform=ax.get_xaxis_transform())
        ax.set_xlabel(xl)
    axes[0].axhline(199.85, color=PAPER_C, lw=1.5, ls="--")
    axes[0].text(0.0, 201, "paper AME on DS", fontsize=7, color=PAPER_C, va="bottom")
    axes[1].axhline(199.85, color=PAPER_C, lw=1.5, ls="--")
    axes[0].set_ylabel("AME on DS (s)")
    save(fig, "sensitivity_lambda_kmax.png")


def jpdc_validation():
    t5 = pd.read_csv(REPRODUCED / "jpdc_validation" / "table5_mean_distance_default.csv")
    t5 = t5[t5["size"] == "all"]
    t4 = pd.read_csv(REPRODUCED / "jpdc_validation" / "table4_transform_order.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=0.55)
    ax = axes[0]
    ax.scatter(t5.paper, t5.ours, s=18, color=OURS_C, zorder=3)
    lim = [0, max(t5.paper.max(), t5.ours.max()) * 1.08]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    for _, r in t5.iterrows():
        if abs(r.ours - r.paper) > 1.0:
            ax.annotate(r.workload, (r.paper, r.ours), fontsize=6.5, color=INK2, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("published (JPDC Table 5)")
    ax.set_ylabel("ours")
    ax.set_title("Mean distance to the workload centroid", fontsize=9)
    ax = axes[1]
    k = t4[t4.algorithm == "kmeans"].copy()
    k["label"] = k["transform"].str.replace("none", "no Cusum").str.replace("cusum", "Cusum") + ", d=" + k["degree"].astype(str)
    y = np.arange(len(k))
    ax.barh(y + 0.19, k.paper_nmi, 0.36, color=PAPER_C, label="paper")
    ax.barh(y - 0.19, k.nmi, 0.36, color=OURS_C, label="ours")
    ax.set_yticks(y, k.label)
    ax.set_xlim(0.6, 1.0)
    ax.set_xlabel("NMI of k-means (k = 24) vs workload labels")
    ax.set_title("JPDC Table 4, k-means", fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "featuriser_vs_jpdc.png")


def scaling():
    f = REPRODUCED / "scaling" / "summary.csv"
    if not f.exists():
        return
    g = pd.read_csv(f)
    g = g[g["mode"] == "local"].sort_values("cores")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    axes[0].plot(g.cores, g.total_s, "-o", color=OURS_C, ms=4)
    axes[0].set_xlabel("cores (Spark local mode)")
    axes[0].set_ylabel("featuriser run time (s)")
    axes[0].set_ylim(0, None)
    axes[1].plot(g.cores, g.speedup, "-o", color=OURS_C, ms=4, label="measured")
    axes[1].plot(g.cores, g.cores / g.cores.min(), "--", color=INK2, lw=1, label="linear")
    axes[1].set_xlabel("cores")
    axes[1].set_ylabel("speed-up")
    axes[1].legend(fontsize=7.5)
    for ax in axes:
        ax.set_xticks(g.cores)
    save(fig, "featuriser_scaling.png")


def reruns():
    d = REPRODUCED / "cluster_rerun"
    if not (d / "summary.json").exists():
        return
    s = json.loads((d / "summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})

    ax = axes[0]
    labels = ["Garralda", "Prats"]
    rr = [s["garralda"]["reruns_top1"], s["prats"]["reruns_top1"]]
    lo = [s["garralda"]["loo_2022_top1_same_workloads"], s["prats"]["loo_2022_top1_same_workloads"]]
    x = np.arange(2)
    ax.bar(x - 0.19, lo, 0.36, color=PAPER_C, label="2022 runs (leave-one-out)")
    ax.bar(x + 0.19, rr, 0.36, color=OURS_C, label="our re-runs vs 2022")
    for i in range(2):
        ax.text(i - 0.19, lo[i] + 0.01, f"{lo[i]:.2f}", ha="center", fontsize=7, color=INK2)
        ax.text(i + 0.19, rr[i] + 0.01, f"{rr[i]:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("nearest run is the same workload")
    ax.set_title("Workload recognition (1-NN)", fontsize=9)
    ax.legend(fontsize=6.8, loc="upper center", ncol=1)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    dg = pd.read_csv(d / "distances_garralda.csv")
    groups = [("VM rep 1\nvs rep 2", dg.to_other_rep.dropna()),
              ("VM vs 2022\nsame setting", dg.to_2022_same_setting.dropna()),
              ("VM vs other\nworkloads", dg.to_2022_other_workloads_median.dropna())]
    ax.boxplot([g[1] for g in groups], widths=0.5, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#f6c8b5", edgecolor=OURS_C), medianprops=dict(color=OURS_C, lw=2),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ref = s["garralda"]["distances_median"]["2022_same_workload_size_other_setting"]
    ax.axhline(ref, color=PAPER_C, ls="--", lw=1.3, label="2022 runs: same workload,\nother setting (median)")
    ax.legend(fontsize=6.8, loc="upper left")
    ax.set_xticks([1, 2, 3], [g[0] for g in groups], fontsize=7)
    ax.set_ylabel("distance (Garralda descriptor)")
    ax.set_title("How far the re-runs land", fontsize=9)
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    t = pd.read_csv(d / "times.csv")
    cells = t.groupby(["workload", "scale", "setting"])[["t_vm", "t_2022"]].mean().reset_index()
    ax.scatter(cells.t_2022, cells.t_vm, s=16, color=OURS_C, zorder=3)
    lim = [2, 400]
    ax.plot(lim, lim, color=INK2, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("2022 run time (s)")
    ax.set_ylabel("re-run time on the VM (s)")
    ax.set_title(f"Run times (Spearman {s['times']['spearman_cells']:.2f})", fontsize=9)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "reruns.png")


def main():
    for f in (ame_bars, seed_spread, sensitivity, jpdc_validation, scaling, reruns):
        try:
            f()
        except FileNotFoundError as e:
            print("skipped", f.__name__, e)


if __name__ == "__main__":
    main()
