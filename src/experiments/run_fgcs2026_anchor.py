"""Re-runs the authors' FGCS 2026 offline model comparison as it is, to check the model code.

The FGCS 2026 repository (mgarralda/spark-self-tuning-framework) comes with 800 runs already described
by the authors' own featuriser, which takes my featuriser out of the picture: if this matches the
published numbers, differences in the FGCS 2025 reproduction can't come from the model code or the
environment. Only two things are swapped: MongoDB for an in-memory mongomock loaded from
resources/dataset/historical_dataset.json, and the missing `parser` package for stubs that this
experiment never calls.

    python src/experiments/run_fgcs2026_anchor.py [garralda|turbo|yoro]     (turbo and yoro take hours)
"""
import contextlib
import enum
import io
import json
import re
import runpy
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external" / "spark-self-tuning-framework"
OUT = ROOT / "results" / "reproduced" / "fgcs2026_anchor"


def install_stubs():
    sys.path[:0] = [str(REPO / "src"), str(REPO / "src_resources")]
    pkg = "framework.proposed.workload_characterization.parser"
    for name in [pkg, pkg + ".feature", pkg + ".eventlog", pkg + ".eventlog_restapi",
                 pkg + ".eventlog_instrumentation", pkg + ".feature_engineering", pkg + ".profile"]:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m

    class _Stub:
        def __init__(self, *a, **k):
            raise RuntimeError("parser not published in the repository")
    f = sys.modules[pkg + ".feature"]
    f.SparkEventLogFeature = f.SparkEventLogFeatureStatistics = _Stub
    f.PolynomialRegressionErrorType = enum.Enum("PolynomialRegressionErrorType", "MSE MAE")
    for mod, cls in [(".eventlog", "SparkEventLog"), (".eventlog_restapi", "SparkEventLogRestApi"),
                     (".eventlog_instrumentation", "SparkEventLogInstrumentation"),
                     (".feature_engineering", "FeatureEngineering"), (".profile", "SparkEventLogProfile")]:
        setattr(sys.modules[pkg + mod], cls, _Stub)
    import mongomock
    import pymongo
    from bson import json_util
    client = mongomock.MongoClient()
    data = json_util.loads((REPO / "resources/dataset/historical_dataset.json").read_text())
    client["spark_event_log"]["historical_dataset"].insert_many(data)
    pymongo.MongoClient = lambda *a, **k: client
    import utils.persistence.mongo.driver as drv
    drv.MongoClient = pymongo.MongoClient
    return len(data)


def parse_blocks(text):
    """model block title -> pooled LOGOCV metrics (first beta block = beta 1.0)."""
    res, title, beta = {}, None, None
    for line in text.splitlines():
        if line.startswith("BETA"):
            beta = line.split("=")[1].strip()
        if "regression model" in line:
            title = line.split(" regression model")[0].strip()
        m = re.search(r"MAE: ([\d.]+) \| RMSE: ([\d.]+) \| P95: ([\d.]+) \| Max: ([\d.]+)", line)
        if m and "micro/pooled" in prev:
            res.setdefault(f"{title or 'model'}|beta={beta}", {}).update(
                dict(zip(["MAE", "RMSE", "P95", "Max"], map(float, m.groups()))))
        m = re.search(r"Kendall tau \(fold-agg\): ([\d.]+) \| Precision@10 \(fold-agg\): ([\d.]+) \| ESR_mu \(fold-avg\): ([\d.]+)", line)
        if m and "micro/pooled" in prev2:
            res.setdefault(f"{title or 'model'}|beta={beta}", {}).update(
                dict(zip(["Kendall_tau", "Precision_at_10", "ESR_mu"], map(float, m.groups()))))
        prev2 = prev if "prev" in dir() else ""
        prev = line
    return res


def main(which="garralda"):
    OUT.mkdir(parents=True, exist_ok=True)
    n = install_stubs()
    runner = REPO / "src_resources" / "runners" / "perfomance_models" / f"{which}_perf_model_runner.py"
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(runner), run_name="__main__")
    secs = time.time() - t0
    text = buf.getvalue()
    (OUT / f"{which}_stdout.txt").write_text(text)
    published_file = {"garralda": "garralda_perf_model_result_N.txt", "turbo": "turbo_perf_model_result.txt",
                      "yoro": "yoro_perf_model_result.txt"}[which]
    pub_text = (REPO / "resources" / "experiment_results" / "performance_model" / published_file).read_text()
    ours = parse_pooled(text)
    theirs = parse_pooled(pub_text)
    cmp = []
    for k, v in ours.items():            # the runner prints one unlabeled block; find the published block it equals
        for kk, vv in theirs.items():
            same = all(abs(v[m] - vv[m]) < 5e-4 for m in v if m in vv)
            if same:
                cmp.append({"our_block": k, "published_block": kk, "ours": v, "published_file": vv,
                            "identical_to_3_decimals": True})
        if not any(c["our_block"] == k for c in cmp):
            cmp.append({"our_block": k, "ours": v, "identical_to_3_decimals": False})
    paper = json.loads((ROOT / "results" / "paper_reference" / "fgcs2026_published.json").read_text())["table5"]
    col = {"garralda": "Ours_Ridge", "turbo": "TurBO", "yoro": "YORO"}[which]
    res_paper = dict(zip(paper["_columns"], paper[col]))
    res = {"runner": runner.name, "executions": n, "seconds": round(secs, 1), "comparison": cmp,
           "paper_table5_rounded": res_paper}
    (OUT / f"{which}_comparison.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def parse_pooled(text):
    """Parse every 'LOGOCV (micro/pooled ...)' block, keyed by '<model title>|beta=<beta>'."""
    out, title, beta, lines = {}, "model", "1.0", text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("BETA"):
            beta = line.split("=")[1].strip()
        if "regression model" in line:
            title = line.split(" regression model")[0].strip()
        if line.strip().startswith("LOGOCV (micro/pooled"):
            nxt = " ".join(lines[i + 1:i + 3])
            m1 = re.search(r"MAE: ([\d.]+) \| RMSE: ([\d.]+) \| P95: ([\d.]+) \| Max: ([\d.]+)", nxt)
            m2 = re.search(r"Kendall tau \(fold-agg\): ([\d.]+) \| Precision@10 \(fold-agg\): ([\d.]+) \| ESR_mu \(fold-avg\): ([\d.]+)", nxt)
            key = f"{title}|beta={beta}"
            if key in out:           # same title twice in one file -> keep order
                key += "#2"
            out[key] = {}
            if m1:
                out[key].update(dict(zip(["MAE", "RMSE", "P95", "Max"], map(float, m1.groups()))))
            if m2:
                out[key].update(dict(zip(["Kendall_tau", "Precision_at_10", "ESR_mu"], map(float, m2.groups()))))
    return out


if __name__ == "__main__":
    main(*(sys.argv[1:]))
