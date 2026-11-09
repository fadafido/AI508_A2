"""Checks the published logs against what the papers say about them.

Counts per setting (JPDC Table 3), the configuration in each log against its folder name, run time
statistics per workload (FGCS Table 3), task counts and input sizes (JPDC Table 2).
Output: results/reproduced/data_audit/
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import INTERIM, PAPER_REF, REPRODUCED  # noqa: E402

OUT = REPRODUCED / "data_audit"
OUT.mkdir(parents=True, exist_ok=True)
UNITS = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}


def parse_size(s):
    m = re.match(r"^\s*([\d.]+)\s*([KMGT]?B)?\s*\((\d+)\)", s.replace("GB(", "GB ("))
    val = float(m.group(1)) * UNITS[m.group(2) or "B"] if m.group(2) else float(m.group(1))
    return val, int(m.group(3))


def main():
    apps = pd.read_csv(INTERIM / "apps.csv")
    fg = json.loads((PAPER_REF / "fgcs2025_published.json").read_text())
    jp = json.loads((PAPER_REF / "jpdc2024_published.json").read_text())
    s = {}
    s["applications"] = int(len(apps))
    s["paper_states"] = 216
    s["spark_versions"] = apps.spark_version.value_counts().to_dict()
    s["master_deploy"] = apps.groupby(["master", "deploy_mode"]).size().rename("n").reset_index().to_dict("records")
    s["dynamic_allocation"] = apps.dynamic_allocation.astype(str).value_counts().to_dict()
    counts = apps.groupby(["scale", "setting_id"]).size().unstack(fill_value=0)
    counts.to_csv(OUT / "counts.csv")
    s["counts_by_scale_setting"] = {str(k): {str(c): int(v) for c, v in r.items()} for k, r in counts.iterrows()}
    s["counts_by_setting"] = apps.setting_id.value_counts().sort_index().to_dict()
    s["paper_counts_by_setting"] = {"1": 48, "2": 72, "3": 72, "4": 24}
    # which sample is missing
    full = {(w, sc, st) for w in apps.workload.unique() for sc, sts in
            {"small": [1, 2, 3], "medium": [1, 2, 3], "large": [2, 3, 4]}.items() for st in sts}
    have = set(zip(apps.workload, apps.scale, apps.setting_id))
    s["missing_samples"] = sorted([list(x) for x in full - have])
    # folder name vs recorded configuration
    mism = []
    for r in apps.itertuples():
        m = re.search(r"executor(\d+)_cores(\d+)_memory(\d+)g_shuffle(\d+)", r.folder)
        named = tuple(int(x) for x in m.groups())
        rec = (r.executor_instances, r.executor_cores, int(round(r.executor_memory_gb)), r.sql_shuffle_partitions)
        if named != rec:
            mism.append({"folder": r.folder, "named": named, "recorded": rec})
    s["folder_vs_recorded_config_mismatches"] = {"n": len(mism), "distinct": sorted({(m["folder"], m["named"], m["recorded"]) for m in mism})}
    s["inner_folder_prefix_by_scale"] = apps.groupby("scale").folder.apply(lambda x: sorted({f.split("_")[0] for f in x})).to_dict()
    s["driver_memory_gb_by_scale"] = apps.groupby("scale").driver_memory_gb.unique().apply(list).to_dict()
    s["failed_tasks"] = int(apps.n_failed_tasks.sum())
    # execution-time statistics vs FGCS 2025 Table 3
    rows = []
    t3 = fg["table3_T_stats_seconds"]
    for w, g in list(apps.groupby("workload")) + [("ALL", apps)]:
        t = g.exec_time_s.values
        ref = t3.get(w)
        row = {"workload": w, "n": len(t), "min": t.min(), "max": t.max(), "mean": t.mean(), "std_pop": t.std(),
               "std_sample": t.std(ddof=1)}
        if ref:
            row.update({"paper_min": ref[0], "paper_max": ref[1], "paper_mean": ref[2], "paper_std": ref[3]})
            d = np.abs(np.array([t.min(), t.max(), t.mean(), t.std()]) - np.array(ref, float))
            rel = d / np.maximum(np.array(ref, float), 1)
            row["max_abs_diff_s"] = float(d.max())
            row["max_rel_diff"] = float(rel.max())
            # consistent = every statistic within 1.5 s (covers any rounding/truncation convention of the paper)
            row["class"] = "consistent" if d.max() <= 1.5 else ("minor (<=5%)" if rel.max() <= 0.05 else "different runs")
        rows.append(row)
    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT / "t_stats_vs_paper.csv", index=False, float_format="%.3f")
    sub = tdf[tdf.workload != "ALL"]
    s["t_stats_vs_fgcs_table3"] = {"workloads": int(len(sub)), "by_class": sub["class"].value_counts().to_dict(),
                                   "different_runs": sub.loc[sub["class"] == "different runs", "workload"].tolist(),
                                   "minor": sub.loc[sub["class"] == "minor (<=5%)", "workload"].tolist(),
                                   "aggregate_std_note": "paper ALL std=576 is inconsistent with its own per-workload rows (pooled std implied ~770); ours (pop) = %.1f" % apps.exec_time_s.std(ddof=0)}
    # task counts and input bytes vs JPDC Table 2
    t2 = jp["table2_input_and_max_tasks"]
    rows = []
    for w, sizes in t2.items():
        for i, sc in enumerate(["small", "medium", "large"]):
            g = apps[(apps.workload == w) & (apps.scale == sc)]
            if g.empty:
                continue
            pv, pt = parse_size(sizes[i])
            rows.append({"workload": w, "scale": sc, "paper_input_bytes": pv, "paper_max_tasks": pt,
                         "our_max_tasks": int(g.n_tasks.max()), "our_tasks_by_setting": g.sort_values("setting_id").n_tasks.tolist(),
                         "our_max_input_bytes": float(g.input_bytes.max()),
                         "tasks_match": int(g.n_tasks.max()) == pt,
                         "input_ratio_ours_to_paper": (float(g.input_bytes.max()) / pv) if pv else None})
    tk = pd.DataFrame(rows)
    tk.to_csv(OUT / "tasks_vs_paper.csv", index=False)
    s["tasks_match"] = {"cells": int(len(tk)), "max_tasks_equal": int(tk.tasks_match.sum()),
                        "not_equal": tk.loc[~tk.tasks_match, ["workload", "scale", "paper_max_tasks", "our_max_tasks"]].to_dict("records")}
    ok = tk.input_ratio_ours_to_paper.dropna()
    s["input_bytes_ratio_ours_to_paper"] = {"median": float(ok.median()), "within_5pct": int(((ok > 0.95) & (ok < 1.05)).sum()),
                                            "cells_with_nonzero_paper_value": int(len(ok))}
    (OUT / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
