"""Runs the FGCS 2025 experiments (Tables 5-12) with the authors' model code on our descriptors.

    python src/experiments/run_fgcs2025.py --check      # our driver vs the authors' validation.py
    python src/experiments/run_fgcs2025.py --datasets DS --models garralda,dou,prats --seeds 0
    python src/experiments/run_fgcs2025.py --datasets DSp,DSpp --models garralda --seeds 0-9
    python src/experiments/run_fgcs2025.py --datasets DS --models garralda --regressors rf,adaboost,gb --seeds 0-2
    python src/experiments/run_fgcs2025.py --datasets DS --models garralda --lambdas 0,0.1,0.2

Results go to results/reproduced/fgcs2025/<dataset>/<model>[_<regressor>][_lam<l>][_<variant>]/seed<k>.json.
With --budget S a run stops after S seconds and carries on from its cache the next time.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments import cv as CV  # noqa: E402
from src.experiments.datasets import build  # noqa: E402

OUT = ROOT / "results" / "reproduced" / "fgcs2025"


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def check_equivalence():
    """Authors' validation.CrossValidation (patched) vs our parallel driver, Garralda model on DS."""
    import numpy as np
    sys.path.insert(0, str(CV.AUTHORS))
    import validation
    from evaluation import EvaluationMetrics
    ds = build("DS")
    v = validation.CrossValidation(w=ds["W1"], w_groups=ds["group"], cs=ds["CS"], t=ds["T"], k_min=3, k_max=18,
                                   eebc_weight=0.5)
    t0 = time.time()
    lo, lg = v.leave_one_out(), v.leave_one_group_out()
    a = {"LOOCV": [lo.MAE(), lo.RMSE(), lo.HME()], "LOGOCV": [lg.MAE(), lg.RMSE(), lg.HME()],
         "AME": EvaluationMetrics.AME(lo.HME(), lg.HME()), "seconds": time.time() - t0}
    r1, r2 = CV.run("garralda", ds, "loocv"), CV.run("garralda", ds, "logocv")
    b = {"LOOCV": [r1["metrics"][k] for k in ("MAE", "RMSE", "HME")],
         "LOGOCV": [r2["metrics"][k] for k in ("MAE", "RMSE", "HME")],
         "AME": (r1["metrics"]["HME"] + r2["metrics"]["HME"]) / 2}
    same = all(np.allclose(a[k], b[k], rtol=0, atol=1e-9) for k in ("LOOCV", "LOGOCV", "AME"))
    res = {"authors_validation_py": a, "our_driver": b, "identical": bool(same)}
    (OUT / "driver_equivalence.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--datasets", default="DS")
    ap.add_argument("--models", default="garralda")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--regressors", default="nnls")
    ap.add_argument("--lambdas", default="0.5")
    ap.add_argument("--variant", default="default", help="descriptor variant (featuriser ablation)")
    ap.add_argument("--kmax", type=int, default=None, help="override k_max (sensitivity)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="seconds per call; resumable via per-unit cache")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check:
        return check_equivalence()
    import sklearn, numpy, scipy
    env = {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
           "numpy": numpy.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
           "cpus": os.cpu_count()}
    for dsn in a.datasets.split(","):
        for seed in parse_seeds(a.seeds):
            ds = build(dsn, seed, a.variant)
            kr = (ds["k_range"][0], a.kmax) if a.kmax else None
            for model in a.models.split(","):
                regs = a.regressors.split(",") if model == "garralda" else ["-"]
                lams = [float(x) for x in a.lambdas.split(",")] if model == "garralda" else [0.5]
                for reg in regs:
                    for lam in lams:
                        tag = model + (f"_{reg}" if reg not in ("nnls", "-") else "") + \
                              (f"_lam{lam:g}" if lam != 0.5 else "") + (f"_{a.variant}" if a.variant != "default" else "") + \
                              (f"_kmax{a.kmax}" if a.kmax else "")
                        f = OUT / dsn / tag / f"seed{seed}.json"
                        if f.exists() and not a.force:
                            print(f"skip {f.relative_to(ROOT)}")
                            continue
                        f.parent.mkdir(parents=True, exist_ok=True)
                        out = {"dataset": dsn, "variant": a.variant, "env": env, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
                        cache = (OUT / "_cache" / dsn / tag / f"seed{seed}") if a.budget else None
                        t_call = time.time()
                        for cvn in ("loocv", "logocv"):
                            left = None if a.budget is None else a.budget - (time.time() - t_call)
                            out[cvn] = CV.run(model, ds, cvn, seed=seed, n_jobs=a.jobs, lam=lam,
                                              regressor=reg if reg != "-" else "nnls", k_range=kr,
                                              cache_dir=(cache / cvn) if cache else None, budget=left)
                        if any(out[c].get("incomplete") for c in ("loocv", "logocv")):
                            print(f"INCOMPLETE {dsn} {tag} seed{seed}: loocv {out['loocv'].get('done', 'all')}/"
                                  f"{out['loocv'].get('total', '-')} logocv {out['logocv'].get('done', 'all')}/"
                                  f"{out['logocv'].get('total', '-')}", flush=True)
                            sys.exit(3)
                        out["AME"] = (out["loocv"]["metrics"]["HME"] + out["logocv"]["metrics"]["HME"]) / 2
                        if "new_types" in ds:
                            out["new_types"] = ds["new_types"]
                        f.write_text(json.dumps(out))
                        m1, m2 = out["loocv"]["metrics"], out["logocv"]["metrics"]
                        print(f"{dsn:5s} {tag:22s} seed{seed} LOOCV MAE {m1['MAE']:8.2f} RMSE {m1['RMSE']:8.2f} HME {m1['HME']:8.2f} | "
                              f"LOGOCV MAE {m2['MAE']:8.2f} RMSE {m2['RMSE']:8.2f} HME {m2['HME']:8.2f} | AME {out['AME']:8.2f} "
                              f"| {out['loocv']['seconds'] + out['logocv']['seconds']:.0f}s errors {out['loocv']['n_errors']}/{out['logocv']['n_errors']}",
                              flush=True)


if __name__ == "__main__":
    main()
