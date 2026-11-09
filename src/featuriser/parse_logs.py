"""Parses Spark event logs into per-application task series and an apps.csv table.

Plain single-machine version of the featuriser (one process per log file). It is the reference for
checking the PySpark version, and it keeps every task so the descriptor variants can be recomputed
quickly without touching the 4 GB of logs again.

    python src/featuriser/parse_logs.py                     # 2022 logs in data/raw/eventlogs
    python src/featuriser/parse_logs.py --logs data/cluster/eventlogs --out data/interim/cluster

Writes <out>/series/<app>.npz (24 task metrics, task/stage ids, launch/finish times, failed flag,
tasks per stage) and <out>/apps.csv (configuration, run time, sizes; one row per app).
"""
import csv
import json
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import LOGS_DIR, INTERIM  # noqa: E402
from src.featuriser.descriptor import task_metric_values, N_TASK_METRICS  # noqa: E402

# (executors, cores, memory GB, shuffle partitions) -> setting number in JPDC Table 3
SETTING_IDS = {(1, 4, 4, 200): 1, (3, 3, 3, 200): 2, (3, 3, 3, 100): 3, (3, 3, 5, 200): 4}


def mem_gb(v):
    if v is None:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)([gmk]?)b?$", str(v).strip().lower())
    if not m:
        return None
    x = float(m.group(1))
    return {"g": x, "m": x / 1024, "k": x / 1024 / 1024, "": x / 1024 ** 3}[m.group(2)]


def parse(path: Path, series_dir: Path) -> dict:
    t0 = time.time()
    scale, folder, app = path.parts[-3], path.parts[-2], path.parts[-1]
    rows, task_id, stage_id, launch, finish, failed = [], [], [], [], [], []
    props, spark_version, t_start, t_end, app_name = {}, None, None, None, None
    jobs, stage_tasks = 0, {}
    n_fail = 0
    with open(path, "rb") as f:
        for line in f:
            head = line[:48]
            if b"SparkListenerTaskEnd" in head:
                e = json.loads(line)
                if not e.get("Task Metrics"):
                    continue                      # no metrics -> not a measurable task event
                ti = e.get("Task Info", {})
                rows.append(task_metric_values(e))
                task_id.append(ti.get("Task ID", -1))
                stage_id.append(e.get("Stage ID", -1))
                launch.append(ti.get("Launch Time", 0))
                finish.append(ti.get("Finish Time", 0))
                fl = bool(ti.get("Failed")) or bool(ti.get("Killed"))
                failed.append(fl)
                n_fail += fl
            elif b"SparkListenerStageCompleted" in head:
                si = json.loads(line).get("Stage Info", {})
                stage_tasks[si.get("Stage ID")] = si.get("Number of Tasks", 0)
            elif b"SparkListenerJobStart" in head:
                jobs += 1
            elif b"SparkListenerApplicationStart" in head:
                e = json.loads(line)
                t_start, app_name = e.get("Timestamp"), e.get("App Name")
            elif b"SparkListenerApplicationEnd" in head:
                t_end = json.loads(line).get("Timestamp")
            elif b"SparkListenerEnvironmentUpdate" in head:
                p = json.loads(line).get("Spark Properties", {})
                props = dict(p) if isinstance(p, list) else p
            elif b"SparkListenerLogStart" in head:
                spark_version = json.loads(line).get("Spark Version")
    m = np.asarray(rows, dtype=np.float64).reshape(-1, N_TASK_METRICS)
    sid = np.asarray(stage_id, dtype=np.int64)
    # tasks per stage, ordered by stage id, counted from the task events actually logged
    st_ids, st_counts = np.unique(sid, return_counts=True) if sid.size else (np.array([], int), np.array([], int))
    np.savez_compressed(series_dir / f"{app}.npz", metrics=m, task_id=np.asarray(task_id, np.int64), stage_id=sid,
                        launch_ms=np.asarray(launch, np.int64), finish_ms=np.asarray(finish, np.int64),
                        failed=np.asarray(failed, bool), stage_ids=st_ids, stage_tasks=st_counts)
    ex_inst = int(props.get("spark.executor.instances", 0) or 0)
    ex_cores = int(props.get("spark.executor.cores", 0) or 0)
    ex_mem = mem_gb(props.get("spark.executor.memory"))
    shuffle = int(props.get("spark.sql.shuffle.partitions", 200) or 200)
    drv_cores = int(props.get("spark.driver.cores", 1) or 1)          # unset in all logs -> Spark default 1
    drv_mem = mem_gb(props.get("spark.driver.memory", "1g"))
    return {
        "app": app, "workload": app.rsplit("_", 1)[-1], "scale": scale, "folder": folder,
        # re-executions use scaled-down memory, their folder is named setting<N> instead
        "setting_id": SETTING_IDS.get((ex_inst, ex_cores, int(round(ex_mem or 0)), shuffle))
                      or (int(folder[7:]) if folder.startswith("setting") else None),
        "spark_version": spark_version, "app_name": app_name, "master": props.get("spark.master"),
        "deploy_mode": props.get("spark.submit.deployMode"),
        "dynamic_allocation": props.get("spark.dynamicAllocation.enabled"),
        "driver_cores": drv_cores, "driver_memory_gb": drv_mem, "executor_cores": ex_cores,
        "executor_instances": ex_inst, "executor_memory_gb": ex_mem, "sql_shuffle_partitions": shuffle,
        "exec_time_s": (t_end - t_start) / 1000.0 if t_start and t_end else None,
        "n_tasks": int(m.shape[0]), "n_failed_tasks": int(n_fail), "n_stages_with_tasks": int(st_ids.size),
        "n_stages_completed": len(stage_tasks), "n_jobs": jobs,
        "input_bytes": float(m[:, 11].sum()) if m.size else 0.0,
        "log_bytes": path.stat().st_size, "parse_seconds": round(time.time() - t0, 3),
    }


def main():
    import argparse
    from functools import partial
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(LOGS_DIR), help="folder with <scale>/<setting>/<app> logs")
    ap.add_argument("--out", default=str(INTERIM))
    ap.add_argument("--budget", type=float, default=1e9, help="stop after this many seconds, rerun to continue")
    a = ap.parse_args()
    logs, out = Path(a.logs), Path(a.out)
    series, rows_dir = out / "series", out / "rows"
    series.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in logs.glob("*/*/*") if p.is_file())
    todo = [p for p in files if not ((rows_dir / f"{p.name}.json").exists() and (series / f"{p.name}.npz").exists())]
    t0 = time.time()
    n = max(1, cpu_count())
    with Pool(n) as pool:
        for i in range(0, len(todo), n):
            if time.time() - t0 > a.budget:
                break
            for r in pool.map(partial(parse, series_dir=series), todo[i:i + n], chunksize=1):
                (rows_dir / f"{r['app']}.json").write_text(json.dumps(r))
    wall = time.time() - t0
    rows = [json.loads((rows_dir / f"{p.name}.json").read_text()) for p in files if (rows_dir / f"{p.name}.json").exists()]
    if len(rows) < len(files):
        print(json.dumps({"parsed": len(rows), "of": len(files), "remaining": len(files) - len(rows), "seconds": round(wall, 1)}))
        return
    rows.sort(key=lambda r: (r["scale"], r["folder"], r["app"]))
    with open(out / "apps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["log_bytes"] for r in rows)
    summary = {"apps": len(rows), "tasks": sum(r["n_tasks"] for r in rows),
               "failed_tasks": sum(r["n_failed_tasks"] for r in rows), "log_GB": round(total / 1e9, 3),
               "cpu_s_sum_of_per_file_parse": round(sum(r["parse_seconds"] for r in rows), 1),
               "workers": cpu_count(), "last_call_wall_s": round(wall, 1)}
    (out / "parse_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
