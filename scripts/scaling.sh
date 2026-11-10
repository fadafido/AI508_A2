#!/usr/bin/env bash
# Featuriser scaling: the same 4.09 GB of logs with local[1], local[2], local[4], REPS times each.
# Each run goes through spark_featurise_chunked.sh (one Spark job per setting folder, timings summed)
# and its descriptors are checked against the plain Python version. Afterwards run
# python src/analysis/scaling_summary.py
#   env: REPS (2), CORES ("1 2 4"), BUDGET (seconds per call, only matters on the Mac VM),
#        LOGS (another copy of the event logs, e.g. on a local disk), PREFIX (run folder prefix,
#        "local" for the shared-mount runs, "vmdisk" for the local-disk runs)
set -uo pipefail
cd "$(dirname "$0")/.."
REPS=${REPS:-2}
CORES=${CORES:-"1 2 4"}
PREFIX=${PREFIX:-local}
export BUDGET=${BUDGET:-100000}
for n in $CORES; do
  for r in $(seq 1 "$REPS"); do
    while true; do
      MASTER="local[$n]" OUT="data/processed/scaling/${PREFIX}${n}_r${r}" bash scripts/spark_featurise_chunked.sh
      rc=$?
      [ $rc -eq 0 ] && break
      [ $rc -eq 3 ] || exit $rc      # 3 = out of time, just run it again
    done
  done
done
