"""Puts our numbers next to the published ones (FGCS 2025 Tables 5-12) and writes the comparison tables.

For models with randomness (random forests in the baselines and the RF/AdaBoost/GB ablation) and for
the synthetic datasets DS' and DS'' we report the mean over the seeds we ran, plus the spread. DS''
has a few seeds where LOGOCV blows up (see docs/issues.md), so for DS'' the median is reported too.

    python src/analysis/compare_with_paper.py
Output: results/comparison/fgcs2025_vs_ours.csv, ame.csv, headline.json
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
            r.update(ours_mean=v.mean(), ours_median=v.median(), ours_sd=v.std(ddof=1) if len(v) > 1 else 0.0,
                     ours_min=v.min(), ours_max=v.max(), ours_seed0=float(v[df.seed == 0].iloc[0]) if (df.seed == 0).any() else np.nan,
                     machines=",".join(sorted(set(df.machine))))
            r["diff_pct_mean"] = 100 * (r["ours_mean"] - p) / p
            r["diff_pct_median"] = 100 * (r["ours_median"] - p) / p
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

    # Table 12 (AME) and the headline claims
    ame = []
    for ds in ("DS", "DSp", "DSpp"):
        for i, model in enumerate(("Prats", "Dou", "Garralda")):
            d = runs(ds, MODEL_DIR[model])
            ame.append({"dataset": ds, "model": model, "paper": PAPER["table12_AME"][DS_NAME[ds]][i],
                        "ours_mean": d.AME.mean() if len(d) else np.nan,
                        "ours_median": d.AME.median() if len(d) else np.nan, "n_seeds": len(d)})
    ame = pd.DataFrame(ame)
    ame.round(2).to_csv(COMPARISON / "ame.csv", index=False)
    head = {"paper": PAPER["headline_claims"]}
    for stat in ("mean", "median"):
        piv = ame.pivot(index="dataset", columns="model", values=f"ours_{stat}")
        if piv.isna().any().any():
            continue
        imp = {b: (100 * (piv[b] - piv["Garralda"]) / piv[b]).round(1).to_dict() for b in ("Prats", "Dou")}
        head[f"ours_{stat}"] = {"improvement_pct_per_dataset": imp,
                                "improvement_pct_avg": {b: round(float(np.mean(list(v.values()))), 1) for b, v in imp.items()}}
    ppiv = ame.pivot(index="dataset", columns="model", values="paper")
    pimp = {b: (100 * (ppiv[b] - ppiv["Garralda"]) / ppiv[b]).round(1).to_dict() for b in ("Prats", "Dou")}
    head["paper_recomputed_from_table12"] = {"improvement_pct_per_dataset": pimp,
                                             "improvement_pct_avg": {b: round(float(np.mean(list(v.values()))), 1)
                                                                     for b, v in pimp.items()}}
    (COMPARISON / "headline.json").write_text(json.dumps(head, indent=2))

    show = df[df.metric.isin(["LOOCV_MAE", "LOGOCV_MAE", "LOOCV_HME", "LOGOCV_HME"])]
    print(show[["table", "dataset", "model", "metric", "paper", "ours_mean", "ours_median", "n_seeds"]]
          .round(1).to_string(index=False))
    print(ame.round(1).to_string(index=False))
    print(json.dumps(head, indent=1))


if __name__ == "__main__":
    main()
