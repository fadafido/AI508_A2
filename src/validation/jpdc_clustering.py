"""Checks the featuriser against the results of the descriptor paper (JPDC 2024).

The authors' featurisation code isn't public, so this compares my version with what JPDC reports for
the same runs:
  Table 4  k-means and agglomerative clustering (k = 24) on min-max scaled descriptors, with and
           without Cusum, polynomial degree 1 to 3
  Table 6  clustering per input size, Garralda against Prats descriptor
  Table 5  mean euclidean distance per workload and size, and the diameter (27.55, Sec. 6.2.2)
The Table 5 distances also decided the task order (D-02 in docs/decisions.md).
Output: results/reproduced/jpdc_validation/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PROCESSED, PAPER_REF, REPRODUCED  # noqa: E402

OUT = REPRODUCED / "jpdc_validation"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(20)


def purity(c, g):
    ct = pd.crosstab(c, g).values
    return ct.max(axis=1).sum() / ct.sum()


def f_measure(c, g, beta=1.0):
    p, ip = purity(c, g), purity(g, c)          # JPDC Eqs. 14-16
    return (1 + beta ** 2) * p * ip / (beta ** 2 * p + ip)


def scores(X, y, algo, seed=0):
    Xs = MinMaxScaler().fit_transform(X)
    if algo == "kmeans":
        lab = KMeans(n_clusters=24, init="k-means++", n_init=10, random_state=seed).fit_predict(Xs)
    else:
        lab = AgglomerativeClustering(n_clusters=24, linkage="ward").fit_predict(Xs)
    return silhouette_score(Xs, lab), f_measure(lab, y), normalized_mutual_info_score(y, lab)


def cluster_table(X, y):
    out = {}
    km = np.array([scores(X, y, "kmeans", s) for s in SEEDS])
    out["kmeans_mean"], out["kmeans_sd"] = km.mean(0), km.std(0)
    out["kmeans_best_f"] = km[km[:, 1].argmax()]
    out["agglomerative"] = np.array(scores(X, y, "agglomerative"))
    return out


def mean_ed(X):
    return 0.0 if len(X) < 2 else float(pdist(X).mean())


def main():
    apps = pd.read_csv(PROCESSED / "apps.csv")
    ref = json.loads((PAPER_REF / "jpdc2024_published.json").read_text())
    y = apps.workload.values
    load = lambda v: pd.read_csv(PROCESSED / f"{v}.csv").set_index("app").loc[apps.app].values
    summary = {"n_points": int(len(apps)), "seeds_kmeans": len(SEEDS)}

    # Table 4 - transformation x polynomial order
    rows = []
    t4 = ref["table4_clustering_all_points"]
    for trans, deg, var in [("none", 1, "garralda__no_cusum_d1"), ("none", 2, "garralda__no_cusum_d2"),
                            ("none", 3, "garralda__no_cusum_d3"), ("cusum", 1, "garralda__cusum_d1"),
                            ("cusum", 2, "garralda__default"), ("cusum", 3, "garralda__cusum_d3")]:
        r = cluster_table(load(var), y)
        for algo, vals, sd in [("kmeans", r["kmeans_mean"], r["kmeans_sd"]), ("agglomerative", r["agglomerative"], [0, 0, 0])]:
            p = t4[f"{algo}|{trans}|{deg}"]
            rows.append({"algorithm": algo, "transform": trans, "degree": deg, "silhouette": vals[0], "silhouette_sd": sd[0],
                         "f_measure": vals[1], "f_measure_sd": sd[1], "nmi": vals[2], "nmi_sd": sd[2],
                         "paper_silhouette": p[0], "paper_f_measure": p[1], "paper_nmi": p[2]})
    t4df = pd.DataFrame(rows)
    t4df.to_csv(OUT / "table4_transform_order.csv", index=False, float_format="%.4f")
    for c in ["silhouette", "f_measure", "nmi"]:
        summary[f"table4_mean_abs_diff_{c}"] = float((t4df[c] - t4df[f"paper_{c}"]).abs().mean())
    # does our implementation reproduce the paper's choice (Cusum + quadratic best on external metrics)?
    km = t4df[t4df.algorithm == "kmeans"].set_index(["transform", "degree"])
    summary["best_kmeans_f_measure_config"] = list(km.f_measure.idxmax())
    summary["best_kmeans_nmi_config"] = list(km.nmi.idxmax())

    # Table 6 - per input size, Garralda vs Prats
    rows = []
    G, P = load("garralda__default"), load("prats")
    t6 = ref["table6_clustering_by_size"]
    for size in ["small", "medium", "large", "all"]:
        m = np.ones(len(apps), bool) if size == "all" else (apps.scale.values == size)
        rg, rp = cluster_table(G[m], y[m]), cluster_table(P[m], y[m])
        for algo in ["kmeans", "agglomerative"]:
            g = rg["kmeans_mean"] if algo == "kmeans" else rg["agglomerative"]
            pp = rp["kmeans_mean"] if algo == "kmeans" else rp["agglomerative"]
            ref6 = t6[f"{algo}|{size}"]
            rows.append({"algorithm": algo, "size": size, "n": int(m.sum()),
                         "g_silhouette": g[0], "g_f_measure": g[1], "g_nmi": g[2],
                         "p_silhouette": pp[0], "p_f_measure": pp[1], "p_nmi": pp[2],
                         "paper_g_silhouette": ref6[0], "paper_g_f_measure": ref6[1], "paper_g_nmi": ref6[2],
                         "paper_p_silhouette": ref6[3], "paper_p_f_measure": ref6[4], "paper_p_nmi": ref6[5]})
    t6df = pd.DataFrame(rows)
    t6df.to_csv(OUT / "table6_by_size.csv", index=False, float_format="%.4f")
    summary["table6_garralda_beats_prats_on_external_metrics"] = {
        "ours": int(((t6df.g_f_measure > t6df.p_f_measure) & (t6df.g_nmi > t6df.p_nmi)).sum()),
        "paper": int(((t6df.paper_g_f_measure > t6df.paper_p_f_measure) & (t6df.paper_g_nmi > t6df.paper_p_nmi)).sum()),
        "of": int(len(t6df))}

    # Table 5 and the diameter for each task order (D-02)
    t5 = ref["table5_mean_euclidean_distance"]
    rows = []
    for var in ["garralda__default", "garralda__order_log", "garralda__order_finish", "garralda__no_final_minmax"]:
        X = load(var)
        diam = float(pdist(X).max())
        diffs = []
        for w, vals in [(k, v) for k, v in t5.items() if not k.startswith("_")]:
            for i, size in enumerate(["small", "medium", "large", "all"]):
                m = (y == w) & ((apps.scale.values == size) if size != "all" else True)
                ours = mean_ed(X[m])
                diffs.append(abs(ours - vals[i]))
                if var == "garralda__default":
                    rows.append({"workload": w, "size": size, "ours": ours, "paper": vals[i]})
        summary[f"{var}"] = {"diameter": diam, "paper_diameter": ref["diameter_FPP"],
                             "table5_mean_abs_diff": float(np.mean(diffs)), "table5_median_abs_diff": float(np.median(diffs))}
    t5df = pd.DataFrame(rows)
    t5df.to_csv(OUT / "table5_mean_distance_default.csv", index=False, float_format="%.3f")
    summary["table5_spearman_default"] = float(t5df[["ours", "paper"]].corr(method="spearman").iloc[0, 1])
    summary["scan_mean_distance_ours"] = t5df[t5df.workload == "scan"].ours.tolist()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(t4df[["algorithm", "transform", "degree", "silhouette", "paper_silhouette", "f_measure", "paper_f_measure", "nmi", "paper_nmi"]].round(3).to_string(index=False))
    print(t6df[["algorithm", "size", "g_f_measure", "paper_g_f_measure", "g_nmi", "paper_g_nmi", "p_f_measure", "paper_p_f_measure", "p_nmi", "paper_p_nmi"]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items()}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
