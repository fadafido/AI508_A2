"""Compares the PySpark featuriser output with the plain Python one (same descriptors, task counts, sizes).

    python src/featuriser/crosscheck.py [spark_output_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PROCESSED, REPRODUCED  # noqa: E402


def main(spark_dir=PROCESSED / "spark"):
    ref_g = pd.read_csv(PROCESSED / "garralda__default.csv").set_index("app").sort_index()
    ref_p = pd.read_csv(PROCESSED / "prats.csv").set_index("app").sort_index()
    ref_a = pd.read_csv(PROCESSED / "apps.csv").set_index("app").sort_index()
    sg = pd.read_csv(Path(spark_dir) / "garralda_spark.csv").set_index("app").sort_index()
    sp = pd.read_csv(Path(spark_dir) / "prats_spark.csv").set_index("app").sort_index()
    sa = pd.read_csv(Path(spark_dir) / "apps_spark.csv").set_index("app").sort_index()
    dg = np.abs(sg.values - ref_g.loc[sg.index].values)
    rel_p = np.abs(sp.values - ref_p.loc[sp.index].values) / np.maximum(np.abs(ref_p.loc[sp.index].values), 1e-12)
    res = {
        "apps_spark": int(len(sg)), "apps_reference": int(len(ref_g)), "same_app_set": set(sg.index) == set(ref_g.index),
        "garralda_max_abs_diff": float(dg.max()), "garralda_mean_abs_diff": float(dg.mean()),
        "garralda_cells_diff_gt_1e-6": int((dg > 1e-6).sum()), "garralda_cells": int(dg.size),
        "prats_max_rel_diff": float(rel_p.max()),
        "n_tasks_equal": bool((sa.n_tasks.values == ref_a.loc[sa.index].n_tasks.values).all()),
        "exec_time_max_abs_diff_s": float(np.abs(sa.exec_time_s.values - ref_a.loc[sa.index].exec_time_s.values).max()),
        "input_bytes_equal": bool(np.allclose(sa.input_bytes.values, ref_a.loc[sa.index].input_bytes.values)),
    }
    (Path(spark_dir) / "crosscheck.json").write_text(json.dumps(res, indent=2))
    if Path(spark_dir).resolve() == (PROCESSED / "spark").resolve():
        (REPRODUCED / "featuriser_crosscheck.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(*(sys.argv[1:]))
