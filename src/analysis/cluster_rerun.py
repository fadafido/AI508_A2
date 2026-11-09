"""Compares the HiBench re-runs on our small cluster with the original 2022 runs.

1. Workload recognition: is the nearest 2022 run (same min-max scaling + euclidean distance as the
   model's kNN) the same workload? For the Garralda and the Prats descriptor, with leave-one-out on
   the 2022 runs as the reference.
2. Distances: re-run vs the 2022 run of the same workload/size/setting, compared with two re-runs of
   the same job (rep 1 vs 2) and with other workloads.
3. Which descriptor features change most between the two platforms.
4. Run times: VM vs 2022 for the same workload/size/setting.
5. The performance model on the re-runs: which neighbours it picks from the 2022 history and how far
   off the predictions are with history = 2022 runs / 2022 + VM runs of the other workloads / only
   VM runs of the other workloads.

    python src/analysis/cluster_rerun.py [--cluster cluster2]
--cluster names the campaign folder (data/<cluster>, data/processed/<cluster>).
Output: results/reproduced/cluster_rerun/ (campaign 2: cluster_rerun2/)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import DATA, PROCESSED, REPRODUCED  # noqa: E402
from src.experiments.datasets import CS_COLS  # noqa: E402
from src.featuriser.descriptor import FEATURE_NAMES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cluster", default="cluster", help="campaign folder under data/ and data/processed/")
ARGS = ap.parse_args()
CLUSTER = ARGS.cluster
OUT = REPRODUCED / ("cluster_rerun" + CLUSTER.removeprefix("cluster"))
# the settings as in the paper; the VM ran them with a smaller executor memory (scripts/cluster/env.sh)
NOMINAL = {1: (1, 1.0, 4, 1, 4.0, 200), 2: (1, 1.0, 3, 3, 3.0, 200), 3: (1, 1.0, 3, 3, 3.0, 100)}


def load(folder):
    apps = pd.read_csv(folder / "apps.csv")
    g = pd.read_csv(folder / "garralda__default.csv").set_index("app").loc[apps.app].values.astype(float)
    p = pd.read_csv(folder / "prats.csv").set_index("app").loc[apps.app].values.astype(float)
    return apps, g, p


def load_reruns():
    apps, g, p = load(PROCESSED / CLUSTER)
    runs = pd.DataFrame([json.loads(x) for x in open(DATA / CLUSTER / "runs.jsonl")])
    runs = runs[runs.rc == 0].drop_duplicates("tag", keep="last").copy()
    runs["app"] = runs["app"] + "_" + runs["workload"]
    apps = apps.merge(runs[["app", "tag", "rep", "setting", "wall_s"]], on="app", how="left")
    keep = apps.rep.notna().values
    apps = apps[keep].reset_index(drop=True)
    apps["rep"] = apps.rep.astype(int)
    apps["setting"] = apps.setting.astype(int)
    return apps, g[keep], p[keep]


def dist(a, b):
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def recognition(D, wl_query, sc_query, wl_ref, sc_ref):
    rows = []
    for i in range(D.shape[0]):
        o = np.argsort(D[i], kind="stable")
        same = wl_ref[o] == wl_query[i]
        rows.append({"nn_workload": wl_ref[o[0]], "top1": bool(same[0]), "hit3": bool(same[:3].any()),
                     "rank_first_same": int(np.argmax(same)) + 1 if same.any() else None,
                     "top1_same_size": bool(same[0] and sc_ref[o[0]] == sc_query[i]),
                     "top1_size": bool(sc_ref[o[0]] == sc_query[i]),
                     "d_nn": float(D[i, o[0]])})
    return pd.DataFrame(rows)


def loo_recognition(Xo, wl_o, sc_o):
    """Leave-one-out self-check on the 2022 runs: the scaler is refitted without the held-out run,
    so the query doesn't influence its own scaling."""
    rows = []
    for i in range(len(Xo)):
        rest = np.delete(np.arange(len(Xo)), i)
        sc = MinMaxScaler().fit(Xo[rest])
        q, G = sc.transform(Xo[i:i + 1]), sc.transform(Xo[rest])
        D = dist(q, G)
        rows.append(recognition(D, wl_o[i:i + 1], sc_o[i:i + 1], wl_o[rest], sc_o[rest]).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o_apps, o_g, o_p = load(PROCESSED)
    r_apps, r_g, r_p = load_reruns()
    wl_o, sc_o = o_apps.workload.values, o_apps.scale.values
    wl_r, sc_r = r_apps.workload.values, r_apps.scale.values
    summary = {"n_original": len(o_apps), "n_reruns": len(r_apps),
               "reruns_per_workload": r_apps.groupby(["workload", "scale"]).size().to_dict()}
    summary["reruns_per_workload"] = {f"{k[0]}/{k[1]}": int(v) for k, v in summary["reruns_per_workload"].items()}
    # only the 2022 runs of workloads we re-ran, for the like-for-like reference numbers
    rerun_wl = sorted(set(wl_r))

    per_run = r_apps[["app", "tag", "workload", "scale", "setting", "rep", "exec_time_s"]].copy()
    for name, Xo, Xr in (("garralda", o_g, r_g), ("prats", o_p, r_p)):
        sc = MinMaxScaler().fit(Xo)
        A, B = sc.transform(Xo), sc.transform(Xr)
        rec = recognition(dist(B, A), wl_r, sc_r, wl_o, sc_o)
        loo = loo_recognition(Xo, wl_o, sc_o)
        loo_sub = loo[np.isin(wl_o, rerun_wl)]
        for c in rec.columns:
            per_run[f"{name}_{c}"] = rec[c].values
        summary[name] = {
            "reruns_top1": round(rec.top1.mean(), 3), "reruns_hit3": round(rec.hit3.mean(), 3),
            "reruns_top1_same_size": round(rec.top1_same_size.mean(), 3),
            "reruns_top1_size": round(rec.top1_size.mean(), 3),
            "reruns_median_rank_first_same": float(np.nanmedian(rec.rank_first_same.astype(float))),
            "loo_2022_top1_all": round(loo.top1.mean(), 3),
            "loo_2022_top1_same_workloads": round(loo_sub.top1.mean(), 3),
            "loo_2022_hit3_same_workloads": round(loo_sub.hit3.mean(), 3),
            "loo_2022_top1_size_all": round(loo.top1_size.mean(), 3),
            "by_workload_top1": rec.groupby(wl_r).top1.mean().round(2).to_dict(),
            "confused_with": rec[~rec.top1].groupby(wl_r[~rec.top1.values]).nn_workload
                             .agg(lambda s: s.value_counts().to_dict()).to_dict(),
        }
        pw = rec.assign(workload=wl_r, scale=sc_r).groupby(["workload", "scale"]).agg(
            n=("top1", "size"), top1=("top1", "mean"), hit3=("hit3", "mean"), top1_size=("top1_size", "mean"))
        pw.round(3).to_csv(OUT / f"per_workload_{name}.csv")

        # distances in the scaled space
        rows = []
        for i, r in r_apps.iterrows():
            m = (wl_o == r.workload) & (sc_o == r.scale)
            cell = np.where(m & (o_apps.setting_id.values == r.setting))[0]
            twin = r_apps[(r_apps.workload == r.workload) & (r_apps.scale == r.scale) &
                          (r_apps.setting == r.setting) & (r_apps.rep != r.rep)].index
            rows.append({
                "app": r.app,
                "to_2022_same_setting": float(np.linalg.norm(B[i] - A[cell[0]])) if len(cell) else np.nan,
                "to_2022_same_workload_size": float(np.mean([np.linalg.norm(B[i] - A[j]) for j in np.where(m)[0]])),
                "to_other_rep": float(np.linalg.norm(B[i] - B[twin[0]])) if len(twin) else np.nan,
                "to_2022_other_workloads_median": float(np.median([np.linalg.norm(B[i] - A[j])
                                                                   for j in np.where(wl_o != r.workload)[0]])),
            })
        d = pd.DataFrame(rows)
        # reference: 2022 runs, same workload and size but a different setting
        cfg = []
        for w in rerun_wl:
            for s in set(sc_r[wl_r == w]):
                idx = np.where((wl_o == w) & (sc_o == s))[0]
                cfg += [np.linalg.norm(A[a] - A[b]) for k, a in enumerate(idx) for b in idx[k + 1:]]
        summary[name]["distances_median"] = {c: round(float(d[c].median()), 3) for c in d.columns if c != "app"}
        summary[name]["distances_median"]["2022_same_workload_size_other_setting"] = round(float(np.median(cfg)), 3)
        d.to_csv(OUT / f"distances_{name}.csv", index=False)

        if name == "garralda":
            # per-feature change (sum of |diff| over the 4 values of each feature), averaged over runs
            drift, noise = [], []
            for i, r in r_apps.iterrows():
                cell = np.where((wl_o == r.workload) & (sc_o == r.scale) & (o_apps.setting_id.values == r.setting))[0]
                if len(cell):
                    drift.append(np.abs(B[i] - A[cell[0]]).reshape(25, 4).sum(1))
                twin = r_apps[(r_apps.workload == r.workload) & (r_apps.scale == r.scale) &
                              (r_apps.setting == r.setting) & (r_apps.rep > r.rep)].index
                for t in twin:
                    noise.append(np.abs(B[i] - B[t]).reshape(25, 4).sum(1))
            fd = pd.DataFrame({"feature": FEATURE_NAMES, "vm_vs_2022": np.mean(drift, 0),
                               "vm_rep1_vs_rep2": np.mean(noise, 0) if noise else np.nan})
            fd.sort_values("vm_vs_2022", ascending=False).to_csv(OUT / "feature_drift.csv", index=False)

    # run times, same workload/size/setting
    t = []
    for _, r in r_apps.iterrows():
        cell = o_apps[(o_apps.workload == r.workload) & (o_apps.scale == r.scale) & (o_apps.setting_id == r.setting)]
        if len(cell):
            t.append({"workload": r.workload, "scale": r.scale, "setting": r.setting, "rep": r.rep,
                      "t_vm": r.exec_time_s, "t_2022": float(cell.exec_time_s.iloc[0])})
    t = pd.DataFrame(t)
    t["ratio"] = t.t_vm / t.t_2022
    t.to_csv(OUT / "times.csv", index=False)
    cells = t.groupby(["workload", "scale", "setting"])[["t_vm", "t_2022"]].mean()
    rho = spearmanr(cells.t_vm, cells.t_2022)
    summary["times"] = {"median_ratio_vm_over_2022": round(float(t.ratio.median()), 3),
                        "spearman_cells": round(float(rho.statistic), 3), "spearman_p": float(rho.pvalue),
                        "n_cells": int(len(cells))}

    summary["model"] = model_on_reruns(o_apps, o_g, r_apps, r_g)
    per_run.to_csv(OUT / "per_run.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if k != "reruns_per_workload"}, indent=1, default=float))


def model_on_reruns(o_apps, o_g, r_apps, r_g, k_range=(3, 18), lam=0.5):
    """The authors' unsupervised + NNLS stages, one re-run at a time. The configuration vector of a
    re-run is either the paper's nominal setting (nominal_cs) or the settings the VM really ran with,
    read from its event log (actual_cs, mainly a smaller executor memory)."""
    from src.experiments.cv import _authors
    algorithm = _authors()[0]
    warnings.filterwarnings("ignore")
    o_cs = o_apps[CS_COLS].values.astype(float)
    o_t, r_t = o_apps.exec_time_s.values.astype(float), r_apps.exec_time_s.values.astype(float)
    variants = {"nominal_cs": np.array([NOMINAL[s] for s in r_apps.setting], float),
                "actual_cs": r_apps[CS_COLS].values.astype(float)}

    def other(i):
        return np.where(r_apps.workload.values != r_apps.workload.values[i])[0]

    out, rows = {}, []
    for vname, r_cs in variants.items():
        histories = {
            "2022": lambda i: (o_g, o_t, o_cs, o_apps.workload.values),
            "2022+vm_other": lambda i: (np.vstack([o_g, r_g[other(i)]]), np.r_[o_t, r_t[other(i)]],
                                        np.vstack([o_cs, r_cs[other(i)]]),
                                        np.r_[o_apps.workload.values, r_apps.workload.values[other(i)]]),
            "vm_other": lambda i: (r_g[other(i)], r_t[other(i)], r_cs[other(i)], r_apps.workload.values[other(i)]),
        }
        out[vname] = {}
        for name, h in histories.items():
            pred, purity, ks = [], [], []
            for i in range(len(r_apps)):
                W, T, CS, wl = h(i)
                kmax = min(k_range[1], len(T) - 1)
                try:
                    idx = algorithm.UnsupervisedStage().get_optimal_insightful_neighbors(W, T, r_g[i], k_range[0], kmax, lam)
                    reg = algorithm.SupervisedStage().get_non_negative_least_squares_regressor_model(X=CS[idx], y=T[idx])
                    p = float(np.ravel(reg.predict(r_cs[i].reshape(1, -1)))[0])
                except Exception:       # same as the authors' CV loop: drop the sample
                    idx, p = np.array([], int), np.nan
                pred.append(p)
                purity.append(float(np.mean(wl[idx] == r_apps.workload.values[i])) if len(idx) else np.nan)
                ks.append(len(idx))
            pred = np.array(pred)
            err = np.abs(pred - r_t)
            out[vname][name] = {"MAE": round(float(np.nanmean(err)), 2), "median_AE": round(float(np.nanmedian(err)), 2),
                                "MAPE_pct": round(float(np.nanmean(err / r_t) * 100), 1),
                                "neighbour_same_workload": round(float(np.nanmean(purity)), 3),
                                "mean_k": round(float(np.mean(ks)), 1), "n_failed": int(np.isnan(pred).sum())}
            rows.append(pd.DataFrame({"app": r_apps.app, "cs": vname, "history": name, "t_true": r_t, "t_pred": pred,
                                      "neighbour_same_workload": purity, "k": ks}))
    pd.concat(rows).to_csv(OUT / "model_predictions.csv", index=False)
    return out


if __name__ == "__main__":
    main()
