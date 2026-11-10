"""Puts our numbers next to the published ones (FGCS 2025 Tables 5-12) and writes the comparison tables.

Aggregation rule, the same for every model, dataset and metric: each metric is computed per seed
(AME included), and "ours" is the median over the seeds we ran. The mean, min, max and sd are kept
as extra columns (sd is not estimated, NaN, when there is one seed). The median is used because a
few DS'' seeds blow up in LOGOCV (I-18) and would dominate any mean; using it everywhere keeps the
rule uniform. Coverage = finite predictions / all samples, averaged over seeds, is reported per CV.

    python src/analysis/compare_with_paper.py
Output: results/comparison/fgcs2025_vs_ours.csv, ame.csv, matched_seeds.csv, headline.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import COMPARISON, PAPER_REF, REPRODUCED  # noqa: E402

RES = REPRODUCED / "fgcs2025"
PAPER = json.loads((PAPER_REF / "fgcs2025_published.json").read_text())
MODEL_DIR = {"Garralda": "garralda", "Dou": "dou", "Prats": "prats"}
DS_NAME = {"DS": "DS", "DSp": "DS_prime", "DSpp": "DS_doubleprime"}


def coverage(cv):
    y = np.asarray(cv["predictions"]["y_pred"], float)
    return float(np.isfinite(y).sum()) / cv["n_samples"]


def runs(ds, tag):
    out = []
    for f in sorted((RES / ds / tag).glob("seed*.json")):
        d = json.loads(f.read_text())
        m1, m2 = d["loocv"]["metrics"], d["logocv"]["metrics"]
        out.append({"seed": int(f.stem[4:]), "LOOCV_MAE": m1["MAE"], "LOOCV_RMSE": m1["RMSE"], "LOOCV_HME": m1["HME"],
                    "LOGOCV_MAE": m2["MAE"], "LOGOCV_RMSE": m2["RMSE"], "LOGOCV_HME": m2["HME"], "AME": d["AME"],
                    "LOOCV_max": m1["residual_max"], "LOOCV_median": m1["residual_median"],
                    "LOOCV_std": m1["residual_std"], "LOGOCV_max": m2["residual_max"],
                    "LOGOCV_median": m2["residual_median"], "LOGOCV_std": m2["residual_std"],
                    "coverage_loocv": coverage(d["loocv"]), "coverage_logocv": coverage(d["logocv"]),
                    "machine": d["env"]["machine"]})
    df = pd.DataFrame(out)
    if len(df):
        for s in ("max", "median", "std"):    # AMR = mean of the LOOCV and LOGOCV residual statistic
            df[f"AMR_{s}"] = (df[f"LOOCV_{s}"] + df[f"LOGOCV_{s}"]) / 2
    return df


def rows_for(table, ds, model, tag, cols, paper_vals):
    df = runs(ds, tag)
    rows = []
    for c, p in zip(cols, paper_vals):
        r = {"table": table, "dataset": ds, "model": model, "metric": c, "paper": p, "n_seeds": len(df)}
        if len(df):
            v = df[c].astype(float)
            r.update(ours_median=v.median(), ours_mean=v.mean(),
                     ours_sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                     ours_min=v.min(), ours_max=v.max(),
                     ours_seed0=float(v[df.seed == 0].iloc[0]) if (df.seed == 0).any() else np.nan,
                     coverage_loocv=df.coverage_loocv.mean(), coverage_logocv=df.coverage_logocv.mean(),
                     machines=",".join(sorted(set(df.machine))))
            r["diff_pct_median"] = 100 * (r["ours_median"] - p) / p
            r["diff_pct_mean"] = 100 * (r["ours_mean"] - p) / p
        rows.append(r)
    return rows


def main():
    rows = []
    t5 = PAPER["table5_regressors_DS"]
    for model, tag in (("RandomForest", "garralda_rf"), ("AdaBoost", "garralda_adaboost"),
                       ("GradientBoosting", "garralda_gb"), ("NNLS", "garralda")):
        rows += rows_for("5", "DS", model, tag, t5["_columns"], t5[model])
    for table, rtable, ds in (("6", "7", "DS"), ("8", "9", "DSp"), ("10", "11", "DSpp")):
        key = {"DS": "table6_DS", "DSp": "table8_DSprime", "DSpp": "table10_DSdoubleprime"}[ds]
        rkey = {"DS": "table7_residuals_DS", "DSp": "table9_residuals_DSprime",
                "DSpp": "table11_residuals_DSdoubleprime"}[ds]
        for model in ("Prats", "Dou", "Garralda"):
            rows += rows_for(table, ds, model, MODEL_DIR[model], PAPER[key]["_columns"], PAPER[key][model])
            rows += rows_for(rtable, ds, model, MODEL_DIR[model], PAPER[rkey]["_columns"], PAPER[rkey][model])
    df = pd.DataFrame(rows)
    COMPARISON.mkdir(parents=True, exist_ok=True)
    df.round(3).to_csv(COMPARISON / "fgcs2025_vs_ours.csv", index=False)

    # matched seeds: the three models on the same seeds (0-2) so their spread is comparable
    matched = []
    for ds in ("DS", "DSp", "DSpp"):
        for model in ("Prats", "Dou", "Garralda"):
            d = runs(ds, MODEL_DIR[model])
            for _, r in d[d.seed <= 2].iterrows():
                matched.append({"dataset": ds, "model": model, "seed": int(r.seed), "machine": r.machine,
                                "LOOCV_MAE": r.LOOCV_MAE, "LOGOCV_MAE": r.LOGOCV_MAE,
                                "LOOCV_HME": r.LOOCV_HME, "LOGOCV_HME": r.LOGOCV_HME, "AME": r.AME})
    pd.DataFrame(matched).round(3).to_csv(COMPARISON / "matched_seeds.csv", index=False)

    # Table 12 (AME) and the headline claims
    ame = []
    for ds in ("DS", "DSp", "DSpp"):
        for i, model in enumerate(("Prats", "Dou", "Garralda")):
            d = runs(ds, MODEL_DIR[model])
            ame.append({"dataset": ds, "model": model, "paper": PAPER["table12_AME"][DS_NAME[ds]][i],
                        "ours_median": d.AME.median() if len(d) else np.nan,
                        "ours_mean": d.AME.mean() if len(d) else np.nan, "n_seeds": len(d)})
    ame = pd.DataFrame(ame)
    ame.round(2).to_csv(COMPARISON / "ame.csv", index=False)
    head = {"aggregation_rule": "each metric per seed, the median over seeds; same rule everywhere",
            "paper": PAPER["headline_claims"]}
    for stat in ("median", "mean"):
        piv = ame.pivot(index="dataset", columns="model", values=f"ours_{stat}")
        if piv.isna().any().any():
            continue
        imp = {b: (100 * (piv[b] - piv["Garralda"]) / piv[b]).round(1).to_dict() for b in ("Prats", "Dou")}
        head[f"ours_{stat}"] = {"improvement_pct_per_dataset": imp,
                                "improvement_pct_avg": {b: round(float(np.mean(list(v.values()))), 1) for b, v in imp.items()}}
    # the same improvements on the matched seeds only (0-2, shared by all three models), so the
    # Garralda DS'' value is not helped by the 7 extra seeds the baselines don't have
    m02 = []
    for ds in ("DS", "DSp", "DSpp"):
        for model in ("Prats", "Dou", "Garralda"):
            d = runs(ds, MODEL_DIR[model])
            d = d[d.seed <= 2]
            m02.append({"dataset": ds, "model": model, "AME": d.AME.median() if len(d) else np.nan})
    piv02 = pd.DataFrame(m02).pivot(index="dataset", columns="model", values="AME")
    if not piv02.isna().any().any():
        imp02 = {b: (100 * (piv02[b] - piv02["Garralda"]) / piv02[b]).round(1).to_dict() for b in ("Prats", "Dou")}
        head["ours_matched_seeds_0-2"] = {
            "AME_median_per_dataset": {m: {ds: round(float(v), 2) for ds, v in piv02[m].items()} for m in piv02.columns},
            "improvement_pct_per_dataset": imp02,
            "improvement_pct_avg": {b: round(float(np.mean(list(v.values()))), 1) for b, v in imp02.items()}}

    ppiv = ame.pivot(index="dataset", columns="model", values="paper")
    pimp = {b: (100 * (ppiv[b] - ppiv["Garralda"]) / ppiv[b]).round(1).to_dict() for b in ("Prats", "Dou")}
    head["paper_recomputed_from_table12"] = {"improvement_pct_per_dataset": pimp,
                                             "improvement_pct_avg": {b: round(float(np.mean(list(v.values()))), 1)
                                                                     for b, v in pimp.items()}}
    (COMPARISON / "headline.json").write_text(json.dumps(head, indent=2))

    show = df[df.metric.isin(["LOOCV_MAE", "LOGOCV_MAE", "LOOCV_HME", "LOGOCV_HME"])]
    print(show[["table", "dataset", "model", "metric", "paper", "ours_median", "ours_mean", "n_seeds"]]
          .round(1).to_string(index=False))
    print(ame.round(1).to_string(index=False))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
