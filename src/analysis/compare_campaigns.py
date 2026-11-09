"""Compares re-run campaign 2 (VM 2, executor memory / 2) with campaign 1 (VM 1, / 4).

Campaign 2 re-ran the same plan with more memory and faster cores, so matched runs (same workload,
profile, setting, repetition) show what the descriptor does when only the resources change.

1. Distance between matched runs of the two campaigns, in the same scaled Garralda space as the
   model's kNN (scaler fitted on the 2022 gallery), against the rep 1 vs rep 2 noise inside each
   campaign and the distance to other workloads.
2. Workload recognition: is the nearest campaign-1 run of each campaign-2 run the same workload?
3. Run times: campaign 2 over campaign 1 for matched runs.
4. Which features move most between the campaigns.

    python src/analysis/compare_campaigns.py [--a cluster --b cluster2]
Output: results/reproduced/cluster_rerun2/vs_campaign1.json (+ matched_pairs.csv, feature_drift_campaigns.csv)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import DATA, PROCESSED, REPRODUCED  # noqa: E402
from src.featuriser.descriptor import FEATURE_NAMES  # noqa: E402


def load_campaign(name):
    apps = pd.read_csv(PROCESSED / name / "apps.csv")
    g = pd.read_csv(PROCESSED / name / "garralda__default.csv").set_index("app").loc[apps.app].values.astype(float)
    runs = pd.DataFrame([json.loads(x) for x in open(DATA / name / "runs.jsonl")])
    runs = runs[runs.rc == 0].drop_duplicates("tag", keep="last").copy()
    runs["app"] = runs["app"] + "_" + runs["workload"]
    apps = apps.merge(runs[["app", "tag", "rep", "setting", "wall_s"]], on="app", how="left")
    keep = apps.rep.notna().values
    apps = apps[keep].reset_index(drop=True)
    return apps, g[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="cluster", help="campaign 1 folder under data/")
    ap.add_argument("--b", default="cluster2", help="campaign 2 folder under data/")
    args = ap.parse_args()
    out = REPRODUCED / ("cluster_rerun" + args.b.removeprefix("cluster"))
    out.mkdir(parents=True, exist_ok=True)

    a_apps, a_g = load_campaign(args.a)
    b_apps, b_g = load_campaign(args.b)
    gallery = pd.read_csv(PROCESSED / "apps.csv")
    g2022 = pd.read_csv(PROCESSED / "garralda__default.csv").set_index("app").loc[gallery.app].values.astype(float)
    sc = MinMaxScaler().fit(g2022)
    A, B = sc.transform(a_g), sc.transform(b_g)

    ia = {t: i for i, t in enumerate(a_apps.tag)}
    pairs = [(ia[t], j) for j, t in enumerate(b_apps.tag) if t in ia]
    rows = []
    for i, j in pairs:
        other = np.where(a_apps.workload.values != b_apps.workload.values[j])[0]
        rows.append({"tag": b_apps.tag[j], "workload": b_apps.workload[j], "scale": b_apps.scale[j],
                     "setting": int(b_apps.setting[j]),
                     "d_matched": float(np.linalg.norm(B[j] - A[i])),
                     "d_other_workloads_median": float(np.median([np.linalg.norm(B[j] - A[k]) for k in other])),
                     "t_c1": float(a_apps.exec_time_s[i]), "t_c2": float(b_apps.exec_time_s[j])})
    d = pd.DataFrame(rows)
    d["t_ratio"] = d.t_c2 / d.t_c1
    d.to_csv(out / "matched_pairs.csv", index=False)

    def rep_noise(apps, X):
        n = []
        for _, r in apps[apps.rep == 1].iterrows():
            twin = apps[(apps.workload == r.workload) & (apps.scale == r.scale) &
                        (apps.setting == r.setting) & (apps.rep == 2)].index
            if len(twin):
                n.append(np.linalg.norm(X[apps.index.get_loc(r.name)] - X[twin[0]]))
        return n

    # recognition: nearest campaign-1 run of each campaign-2 run
    Dba = np.sqrt(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1))
    nn = np.argmin(Dba, 1)
    top1 = a_apps.workload.values[nn] == b_apps.workload.values
    cells = d.groupby(["workload", "scale", "setting"])[["t_c1", "t_c2"]].mean()
    rho = spearmanr(cells.t_c1, cells.t_c2)
    summary = {
        "n_campaign1": len(a_apps), "n_campaign2": len(b_apps), "n_matched": len(pairs),
        "d_matched_median": round(float(d.d_matched.median()), 3),
        "rep_noise_campaign1_median": round(float(np.median(rep_noise(a_apps, A))), 3),
        "rep_noise_campaign2_median": round(float(np.median(rep_noise(b_apps, B))), 3),
        "d_other_workloads_median": round(float(d.d_other_workloads_median.median()), 3),
        "c2_nearest_c1_same_workload": round(float(top1.mean()), 3),
        "times": {"median_ratio_c2_over_c1": round(float(d.t_ratio.median()), 3),
                  "spearman_cells": round(float(rho.statistic), 3), "n_cells": int(len(cells))},
    }

    drift = np.stack([np.abs(B[j] - A[i]).reshape(25, 4).sum(1) for i, j in pairs]).mean(0)
    fd = pd.DataFrame({"feature": FEATURE_NAMES, "c2_vs_c1": drift}).sort_values("c2_vs_c1", ascending=False)
    fd.to_csv(out / "feature_drift_campaigns.csv", index=False)
    summary["features_moving_most"] = fd.feature.head(4).tolist()

    (out / "vs_campaign1.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
