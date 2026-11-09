"""LOOCV and LOGOCV around the authors' model code, run in parallel.

Every prediction comes from the authors' own classes:
  Garralda  algorithm.UnsupervisedStage.get_optimal_insightful_neighbors, then
            SupervisedStage.get_<regressor>_model. These are the two calls inside
            GarraldaPerformanceModel.fit_predict, made separately so k can be logged and the base
            learner swapped for the Table 5 ablation.
  Dou       baselines.dou_perf_model.DouPerformanceModel.predict
  Prats     baselines.prats_perf_model.PratsPerformanceModel.get_random_forest_regressor_model
The metrics come from the authors' evaluation.EvaluationMetrics: LOOCV pooled, LOGOCV per held-out group
and then averaged (Sec. 6.4.1), AME = mean of the two HMEs (Eq. 14).
The authors' random forests have no random_state, so NumPy's global RNG is seeded before every unit of
work (D-11) and several seeds are run.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# AI508_AUTHORS picks another patched copy of the authors' code (the legacy-grid sensitivity of I-09)
AUTHORS = Path(os.environ.get("AI508_AUTHORS", ROOT / "build" / "authors_code")) / "garralda_performance_model"
for p in (str(ROOT), str(AUTHORS)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(AUTHORS), os.environ.get("PYTHONPATH", "")])

from joblib import Parallel, delayed  # noqa: E402


def _authors():
    import algorithm  # noqa: F401
    from baselines import dou_perf_model, prats_perf_model
    from evaluation import EvaluationMetrics
    return algorithm, dou_perf_model, prats_perf_model, EvaluationMetrics


def _predict_unit(model, unit, ds, k_min, k_max, lam, regressor, seed):
    """One unit of work: returns list of (index, y_pred, k_opt)."""
    warnings.filterwarnings("ignore")
    algorithm, dou, prats, _ = _authors()
    train, tests = unit
    np.random.seed((seed * 1_000_003 + int(tests[0])) % (2 ** 32))
    out = []
    try:
        if model == "prats":
            m = prats.PratsPerformanceModel().get_random_forest_regressor_model(
                workload_descriptors=ds["W2"][train], input_data_sizes=ds["ids"][train],
                configuration_settings=ds["CS"][train], execution_times=ds["T"][train])
            X = np.concatenate([ds["W2"][tests], ds["ids"][tests], ds["CS"][tests]], axis=1)
            for i, p in zip(tests, m.predict(X)):
                out.append((int(i), float(p), -1))
            return out
        for i in tests:
            if model == "dou":
                p = dou.DouPerformanceModel().predict(
                    workload_descriptors=ds["W2"][train], configuration_settings=ds["CS"][train],
                    execution_times=ds["T"][train], workload_ref=ds["W2"][i], new_configuration_settings=ds["CS"][i],
                    k=k_max)
                out.append((int(i), float(np.ravel(p)[0]), k_max))
            else:   # garralda
                us, ss = algorithm.UnsupervisedStage(), algorithm.SupervisedStage()
                idx = us.get_optimal_insightful_neighbors(ds["W1"][train], ds["T"][train], ds["W1"][i], k_min, k_max, lam)
                fit = {"nnls": ss.get_non_negative_least_squares_regressor_model,
                       "rf": ss.get_random_forest_regressor_model,
                       "adaboost": ss.get_adaboost_regressor_model,
                       "gb": ss.get_gradientboosting_regressor_model}[regressor]
                reg = fit(X=ds["CS"][train][idx], y=ds["T"][train][idx])
                p = reg.predict(ds["CS"][i].reshape(1, -1))
                out.append((int(i), float(np.ravel(p)[0]), int(len(idx))))
    except Exception as e:   # the authors' CV loops drop failing samples with a warning; we record NaN
        out = [(int(i), float("nan"), -2) for i in tests] if not out else out
        out.append((-1, float("nan"), repr(e)[:200]))
    return out


def units(ds, cv, model):
    n = len(ds["T"])
    allidx = np.arange(n)
    if cv == "loocv":
        return [(np.delete(allidx, i), np.array([i])) for i in range(n)]
    us = []
    for g in sorted(set(ds["group"])):
        test = np.where(ds["group"] == g)[0]
        train = np.where(ds["group"] != g)[0]
        if model == "prats":
            us.append((train, test))                       # one model per held-out group
        else:
            us += [(train, np.array([i])) for i in test]   # lazy model per test sample
    return us


def run(model, ds, cv, seed=0, n_jobs=-1, lam=0.5, regressor="nnls", k_range=None, cache_dir=None, budget=None):
    """Run one cross-validation. With cache_dir, every unit's predictions are stored as they finish, so a run
    can be split over several calls (budget = seconds per call); returns {"incomplete": True, ...} until done."""
    import json as _json
    _, _, _, EM = _authors()
    k_min, k_max = k_range or ds["k_range"]
    t0 = time.time()
    us = units(ds, cv, model)
    if cache_dir is None:
        res = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(_predict_unit)(model, u, ds, k_min, k_max, lam, regressor, seed) for u in us)
    else:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        f = lambda k: cache_dir / f"{cv}_u{k:05d}.json"
        todo = [k for k in range(len(us)) if not f(k).exists()]
        nj = os.cpu_count() if n_jobs in (-1, None) else n_jobs
        with Parallel(n_jobs=n_jobs, batch_size=1) as par:
            for i in range(0, len(todo), max(1, nj) * 2):
                if budget is not None and time.time() - t0 > budget:
                    break
                chunk = todo[i:i + max(1, nj) * 2]
                outs = par(delayed(_predict_unit)(model, us[k], ds, k_min, k_max, lam, regressor, seed) for k in chunk)
                for k, o in zip(chunk, outs):
                    f(k).write_text(_json.dumps(o))
        left = sum(1 for k in range(len(us)) if not f(k).exists())
        if left:
            return {"incomplete": True, "done": len(us) - left, "total": len(us), "seconds": round(time.time() - t0, 1)}
        res = [_json.loads(f(k).read_text()) for k in range(len(us))]
    secs = time.time() - t0
    errors = [r[2] for rr in res for r in rr if r[0] == -1]
    preds = sorted(r for rr in res for r in rr if r[0] >= 0)
    idx = np.array([p[0] for p in preds])
    yp = np.array([p[1] for p in preds])
    kopt = np.array([p[2] for p in preds])
    yt = ds["T"][idx]
    if cv == "loocv":
        em = EM(list(yt), list(yp))
    else:
        groups = ds["group"][idx]
        em = EM.average_metrics([EM(list(yt[groups == g]), list(yp[groups == g])) for g in sorted(set(groups))])
    metrics = {"MAE": em.MAE(), "RMSE": em.RMSE(), "HME": em.HME(), "residual_max": em.residual_max(),
               "residual_median": em.residual_median(), "residual_std": em.residual_std_dev(),
               "residual_mean": em.residual_mean()}
    return {"model": model, "cv": cv, "seed": seed, "lambda": lam, "regressor": regressor, "k_range": [k_min, k_max],
            "n_samples": int(len(ds["T"])), "n_predicted": int(np.isfinite(yp).sum()), "n_errors": len(errors),
            "errors": errors[:5], "seconds": round(secs, 2), "metrics": {k: float(v) for k, v in metrics.items()},
            "predictions": {"index": idx.tolist(), "y_true": yt.tolist(), "y_pred": yp.tolist(), "k_opt": kopt.tolist()}}
