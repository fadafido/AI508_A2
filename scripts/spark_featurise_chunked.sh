#!/usr/bin/env bash
# Runs the PySpark featuriser one setting folder at a time (one Spark job each) and merges the parts.
# I needed this on the Mac VM, where one command can't run for more than 3 minutes. Folders that are
# already done get skipped, so just run it again until it finishes. BUDGET = seconds per call.
#
#   scripts/spark_featurise_chunked.sh                                  # -> data/processed/spark
#   MASTER=local[2] OUT=data/processed/scaling/local2_r1 scripts/spark_featurise_chunked.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/ai508a2/bin/python}
BUDGET=${BUDGET:-150}
MASTER=${MASTER:-local[*]}
OUT=${OUT:-data/processed/spark}
LOGS=${LOGS:-data/raw/eventlogs}
MEM=${SPARK_DRIVER_MEM:-1600m}
export PYSPARK_PYTHON=$PY PYSPARK_DRIVER_PYTHON=$PY
export SPARK_LOCAL_IP=${SPARK_LOCAL_IP:-127.0.0.1} SPARK_LOCAL_HOSTNAME=${SPARK_LOCAL_HOSTNAME:-localhost}
export AI508_SPARK_TMP=${AI508_SPARK_TMP:-$HOME/spark-tmp}
mkdir -p logs
start=$(date +%s)
for d in "$LOGS"/*/*/; do
  rel=${d#"$LOGS"/}; rel=${rel%/}; name=${rel//\//__}
  part=$OUT/chunks/$name
  [ -f "$part/garralda_spark.csv" ] && continue
  if (( $(date +%s) - start > BUDGET - 60 )); then echo "out of time, run again to continue"; exit 3; fi
  echo "[$(date '+%T')] $rel"
  $PY src/featuriser/spark_featuriser.py --master "$MASTER" --driver-memory "$MEM" --logs "$LOGS" \
      --glob "$rel/*" --out "$part" --tag "$name" > "logs/spark_chunk_$name.log" 2>&1 \
    || { echo "failed $rel"; tail -5 "logs/spark_chunk_$name.log"; exit 1; }
  grep -h '"total_s"' "$part/run_$name.json"
done
$PY src/featuriser/merge_spark_chunks.py "$OUT/chunks" "$OUT"
$PY src/featuriser/crosscheck.py "$OUT"
