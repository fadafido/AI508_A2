"""Merges the featuriser output written chunk by chunk (one Spark job per setting folder).

The descriptors of different applications don't depend on each other, so gluing the chunks together
gives the same result as one job over all the logs. Timings are summed over the chunks.

    python src/featuriser/merge_spark_chunks.py <chunks_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PROCESSED  # noqa: E402


def main(chunks_dir=PROCESSED / "spark_chunks", out=PROCESSED / "spark"):
    chunks_dir, out = Path(chunks_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(d for d in chunks_dir.iterdir() if (d / "garralda_spark.csv").exists())
    for name in ["garralda_spark.csv", "prats_spark.csv", "apps_spark.csv"]:
        df = pd.concat([pd.read_csv(d / name) for d in dirs]).sort_values("app")
        df.to_csv(out / name, index=False, float_format="%.17g")
    runs = [json.loads(next(d.glob("run_*.json")).read_text()) for d in dirs]
    tot = {k: round(sum(r["timings"][k] for r in runs), 2) for k in runs[0]["timings"]}
    info = {"tag": out.name, "chunks": len(dirs), "apps": int(sum(r["apps"] for r in runs)),
            "tasks": int(sum(r["tasks"] for r in runs)), "log_bytes": int(sum(r["log_bytes"] for r in runs)),
            "timings": tot, "master": runs[0]["master"], "default_parallelism": runs[0]["default_parallelism"],
            "spark_version": runs[0]["spark_version"]}
    (out / f"run_{out.name}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
