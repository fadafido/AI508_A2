"""Summarises the featuriser scaling runs into results/reproduced/scaling/ (the plot is in figures.py)."""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PROCESSED, REPRODUCED  # noqa: E402


def main():
    rows = []
    for f in sorted((PROCESSED / "scaling").glob("*/run_*.json")):
        d = json.loads(f.read_text())
        tag = d.get("tag") or f.parent.name
        m = re.match(r"(local|standalone_)(\d+)w?_r(\d+)", tag)
        if not m:
            continue
        cc = f.parent / "crosscheck.json"
        ok = None
        if cc.exists():
            try:
                c = json.loads(cc.read_text())
                ok = bool(c["garralda_max_abs_diff"] < 1e-6 and c["n_tasks_equal"])
            except Exception:
                ok = None
        t = d["timings"]
        rows.append({"mode": "local" if m.group(1) == "local" else "standalone",
                     "units": int(m.group(2)), "rep": int(m.group(3)), "cores": d["default_parallelism"],
                     "total_s": t["total_s"], "parse_s": t["parse_tasks_s"], "descriptor_s": t["descriptor_s"],
                     "metadata_s": t["metadata_s"], "prats_s": t["prats_s"], "session_s": t["session_s"],
                     "GB": d["log_bytes"] / 1e9, "tasks": d["tasks"], "matches_reference": ok})
    if not rows:
        print("no scaling runs found")
        return
    df = pd.DataFrame(rows)
    # the fixed part (Spark session start) against the part that can scale with the cores
    df["processing_s"] = df.total_s - df.session_s
    g = df.groupby(["mode", "units", "cores"]).agg(total_s=("total_s", "median"), total_min=("total_s", "min"),
                                                   total_max=("total_s", "max"), reps=("rep", "count"),
                                                   session_s=("session_s", "median"),
                                                   processing_s=("processing_s", "median"),
                                                   GB=("GB", "first"), all_match=("matches_reference", "all")).reset_index()
    for mode in g["mode"].unique():
        base = g[(g["mode"] == mode)].sort_values("cores").iloc[0]
        sel = g["mode"] == mode
        g.loc[sel, "speedup"] = base.total_s / g.loc[sel, "total_s"]
        g.loc[sel, "speedup_processing"] = base.processing_s / g.loc[sel, "processing_s"]
        g.loc[sel, "efficiency"] = g.loc[sel, "speedup"] / (g.loc[sel, "cores"] / base.cores)
    g["MB_per_s"] = g.GB * 1000 / g.total_s
    out = REPRODUCED / "scaling"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "runs.csv", index=False)
    g.to_csv(out / "summary.csv", index=False, float_format="%.3f")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
