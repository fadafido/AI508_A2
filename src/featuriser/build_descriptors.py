"""Builds the Garralda and Prats descriptors from the task series written by parse_logs.py.

Besides the default descriptor it also writes a few variants for the ablation:
  default          tasks in Task ID order, Cusum, final min-max, degree 2 (what we use)
  order_log        tasks in the order they appear in the log
  order_finish     tasks ordered by finish time
  no_cusum_d1/2/3  no Cusum, polynomial degree 1/2/3     (JPDC Table 4, "None" rows)
  cusum_d1/d3      Cusum with degree 1/3                 (JPDC Table 4, "Cusum" rows)
  no_final_minmax  Cusum but no rescaling to [0, 1] afterwards

    python src/featuriser/build_descriptors.py
    python src/featuriser/build_descriptors.py --interim data/interim/cluster --out data/processed/cluster

Output: <out>/garralda__<variant>.csv, <out>/prats.csv, <out>/apps.csv
"""
import argparse
import csv
import json
import sys
import time
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import INTERIM, PROCESSED  # noqa: E402
from src.featuriser.descriptor import garralda_descriptor, prats_descriptor, descriptor_columns  # noqa: E402

VARIANTS = {
    # Task ID order matched the JPDC Table 5 distances best (docs/decisions.md)
    "default": dict(order="task_id", cusum=True, degree=2, final_minmax=True),
    "order_log": dict(order="log", cusum=True, degree=2, final_minmax=True),
    "order_finish": dict(order="finish", cusum=True, degree=2, final_minmax=True),
    "no_cusum_d1": dict(order="task_id", cusum=False, degree=1, final_minmax=True),
    "no_cusum_d2": dict(order="task_id", cusum=False, degree=2, final_minmax=True),
    "no_cusum_d3": dict(order="task_id", cusum=False, degree=3, final_minmax=True),
    "cusum_d1": dict(order="task_id", cusum=True, degree=1, final_minmax=True),
    "cusum_d3": dict(order="task_id", cusum=True, degree=3, final_minmax=True),
    "no_final_minmax": dict(order="task_id", cusum=True, degree=2, final_minmax=False),
}
META_COLS = ["app", "workload", "scale", "setting_id", "driver_cores", "driver_memory_gb", "executor_cores",
             "executor_instances", "executor_memory_gb", "sql_shuffle_partitions", "exec_time_s", "input_bytes",
             "n_tasks"]


def ordered(z, order):
    m = z["metrics"]
    if order == "log":
        return m
    key = {"task_id": z["task_id"], "finish": z["finish_ms"], "launch": z["launch_ms"]}[order]
    return m[np.argsort(key, kind="stable")]


def one(app, series):
    z = np.load(series / f"{app}.npz")
    tps = z["stage_tasks"]          # tasks per stage, by stage id
    out = {"app": app}
    for name, v in VARIANTS.items():
        out[name] = garralda_descriptor(ordered(z, v["order"]), tps, v["degree"], v["cusum"], v["final_minmax"])
    out["prats"] = prats_descriptor(z["metrics"], tps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default=str(INTERIM))
    ap.add_argument("--out", default=str(PROCESSED))
    a = ap.parse_args()
    interim, out = Path(a.interim), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    apps = list(csv.DictReader(open(interim / "apps.csv")))
    t0 = time.time()
    with Pool(max(1, cpu_count())) as pool:
        res = {r["app"]: r for r in pool.map(partial(one, series=interim / "series"), [x["app"] for x in apps])}
    with open(out / "apps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(META_COLS)
        for x in apps:
            w.writerow([x[c] for c in META_COLS])
    for name, v in list(VARIANTS.items()) + [("prats", None)]:
        cols = descriptor_columns("p") if name == "prats" else descriptor_columns("g", v["degree"])
        fname = "prats.csv" if name == "prats" else f"garralda__{name}.csv"
        with open(out / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["app"] + cols)
            for x in apps:
                w.writerow([x["app"]] + [repr(v) for v in res[x["app"]][name]])
    info = {"apps": len(apps), "variants": list(VARIANTS), "seconds": round(time.time() - t0, 1)}
    (out / "descriptors_build.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
