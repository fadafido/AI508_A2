"""Writes the list of cluster jobs (data/cluster/plan.txt) that step.sh works through.

For every profile and workload: generate the data once, then run each setting/repetition in a
shuffled order so slow drift on the machine doesn't line up with one setting.

    python scripts/cluster/make_plan.py --reps tiny=2,small=1
"""
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ["sort", "wordcount", "terasort", "repartition", "sleep", "pagerank", "bayes", "kmeans"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="tiny,small")
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--settings", default="1,2,3")
    ap.add_argument("--reps", default="tiny=2,small=1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "cluster" / "plan.txt"))
    a = ap.parse_args()
    reps = dict(x.split("=") for x in a.reps.split(","))
    rnd = random.Random(a.seed)
    lines = []
    for p in a.profiles.split(","):
        for w in a.workloads.split(","):
            if w != "sleep":
                lines.append(f"prepare {w} {p}")
            runs = [f"run {w} {p} {s} {r}" for s in a.settings.split(",") for r in range(1, int(reps[p]) + 1)]
            rnd.shuffle(runs)
            lines += runs
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {out}")


if __name__ == "__main__":
    main()
