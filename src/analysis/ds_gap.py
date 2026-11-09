"""The DS gap: what is behind our higher Garralda LOOCV error on DS.

1. The worst LOOCV fold (an svm run): its neighbours and k, the prediction in the Yeo-Johnson space
   and back-transformed, and the share of the total squared error that single fold carries.
2. The DS errors rescored by the data-audit class of each workload (I-07): run times consistent with
   FGCS Table 3, different runs, or terasort (the renamed folder).

    python src/analysis/ds_gap.py
Output: results/comparison/ds_gap.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.common.paths import COMPARISON, REPRODUCED  # noqa: E402
from src.experiments.datasets import load_ds  # noqa: E402
from src.experiments.cv import _authors  # noqa: E402

RES = REPRODUCED / "fgcs2025" / "DS"


def worst_fold(ds):
    algorithm = _authors()[0]
    d = json.loads((RES / "garralda" / "seed0.json").read_text())
    pr = d["loocv"]["predictions"]
    idx = np.array(pr["index"], int)
    y_true = np.array(pr["y_true"], float)
    y_pred = np.array(pr["y_pred"], float)
    err = np.abs(y_pred - y_true)
    w = int(np.nanargmax(err))
    i = int(idx[w])
    sq = np.square(y_pred - y_true)
    rmse_all = float(np.sqrt(np.nanmean(sq)))
    rmse_wo = float(np.sqrt(np.nanmean(np.delete(sq, w))))

    # recompute the fold with the authors' two stages, like the CV driver does
    train = np.delete(np.arange(len(ds["T"])), i)
    np.random.seed((0 * 1_000_003 + i) % (2 ** 32))
    k_min, k_max = ds["k_range"]
    nb = algorithm.UnsupervisedStage().get_optimal_insightful_neighbors(
        ds["W1"][train], ds["T"][train], ds["W1"][i], k_min, k_max, 0.5)
    nb = np.asarray(nb, int)
    g = train[nb]
    reg = algorithm.SupervisedStage().get_non_negative_least_squares_regressor_model(
        X=ds["CS"][train][nb], y=ds["T"][train][nb])
    p_back = float(np.ravel(reg.predict(ds["CS"][i].reshape(1, -1)))[0])
    p_trans = float(np.ravel(reg.regressor_.predict(ds["CS"][i].reshape(1, -1)))[0])
    y_trans = reg.transformer_.transform(ds["T"][train][nb].reshape(-1, 1)).ravel()
    return {
        "app": str(ds["app"][i]), "workload": str(ds["group"][i]),
        "y_true": float(y_true[w]), "y_pred_stored": float(y_pred[w]), "y_pred_recomputed": p_back,
        "k": int(len(nb)),
        "neighbours": [{"app": str(ds["app"][j]), "workload": str(ds["group"][j]), "t": float(ds["T"][j])}
                       for j in g],
        "neighbour_t_range": [float(ds["T"][g].min()), float(ds["T"][g].max())],
        "prediction_transformed_space": p_trans,
        "neighbour_targets_transformed_range": [float(y_trans.min()), float(y_trans.max())],
        "loocv_rmse_with_fold": rmse_all, "loocv_rmse_without_fold": rmse_wo,
        "share_of_squared_error_pct": round(100 * float(sq[w] / np.nansum(sq)), 1),
    }


def rescore_by_class(ds):
    audit = pd.read_csv(REPRODUCED / "data_audit" / "t_stats_vs_paper.csv")
    cls = dict(zip(audit.workload, audit["class"]))
    out = {}
    for model in ("garralda", "dou", "prats"):
        per_seed = []
        for f in sorted((RES / model).glob("seed*.json")):
            d = json.loads(f.read_text())
            row = {}
            for cv in ("loocv", "logocv"):
                pr = d[cv]["predictions"]
                idx = np.array(pr["index"], int)
                err = np.abs(np.array(pr["y_pred"], float) - np.array(pr["y_true"], float))
                groups = pd.Series([cls.get(w, "unknown") for w in ds["group"][idx]])
                row.update({f"{cv}_{g}": float(np.nanmean(err[groups.values == g]))
                            for g in sorted(set(groups))})
            per_seed.append(row)
        df = pd.DataFrame(per_seed)
        out[model] = {c: round(float(df[c].median()), 2) for c in df.columns}
    counts = pd.Series([cls.get(w, "unknown") for w in sorted(set(ds["group"]))]).value_counts()
    return {"workloads_per_class": counts.to_dict(), "MAE_by_class_median_over_seeds": out}


def main():
    warnings.filterwarnings("ignore")
    ds = load_ds()
    res = {"worst_loocv_fold": worst_fold(ds), "rescored_by_audit_class": rescore_by_class(ds)}
    COMPARISON.mkdir(parents=True, exist_ok=True)
    (COMPARISON / "ds_gap.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
