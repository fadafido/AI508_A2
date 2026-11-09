#!/usr/bin/env bash
# Full pipeline, in order. Each step writes its own outputs; logs go to logs/run_all_<step>.log
#   PY      python of the main environment      (default: ~/.venvs/ai508a2/bin/python)
#   PY26    python of the FGCS 2026 environment (default: ~/.venvs/ai508a2-fgcs2026/bin/python)
#   STEPS   subset to run, e.g. STEPS="spark crosscheck"
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/ai508a2/bin/python}
PY26=${PY26:-$HOME/.venvs/ai508a2-fgcs2026/bin/python}
export PY
ALL="fetch env patches raw parse descriptors audit validate spark paper_audit anchor experiments scaling reruns compare"
STEPS=${STEPS:-$ALL}
mkdir -p logs
run() { local s=$1; shift; echo "[$(date '+%F %T')] >>> $s"; "$@" > "logs/run_all_$s.log" 2>&1 && echo "[$(date '+%F %T')] ok  $s" || { echo "[$(date '+%F %T')] FAILED $s (see logs/run_all_$s.log)"; return 1; }; }
for s in $STEPS; do case $s in
  fetch)       run fetch bash scripts/fetch_external.sh ;;
  env)         run env env PY=$PY bash env/capture_env.sh "${LABEL:-$(hostname)}" ;;
  patches)     run patches $PY patches/apply_patches.py ;;
  raw)         run raw $PY src/data/prepare_raw.py ;;
  parse)       run parse $PY src/featuriser/parse_logs.py ;;
  descriptors) run descriptors $PY src/featuriser/build_descriptors.py ;;
  audit)       run audit $PY src/data/audit.py ;;
  validate)    run validate $PY src/validation/jpdc_clustering.py ;;
  spark)       run spark env BUDGET=100000 PY=$PY bash scripts/spark_featurise_chunked.sh ;;
  scaling)     run scaling bash -c "PY=$PY bash scripts/scaling.sh && $PY src/analysis/scaling_summary.py" ;;
  reruns)      run reruns bash -c "$PY src/featuriser/parse_logs.py --logs data/cluster/eventlogs --out data/interim/cluster && $PY src/featuriser/build_descriptors.py --interim data/interim/cluster --out data/processed/cluster && $PY src/analysis/cluster_rerun.py" ;;
  paper_audit) run paper_audit $PY src/analysis/audit_paper_tables.py ;;
  anchor)      run anchor $PY26 src/experiments/run_fgcs2026_anchor.py garralda ;;
  experiments) run experiments bash scripts/run_experiments.sh ;;
  compare)     run compare bash -c "$PY src/analysis/compare_with_paper.py && $PY src/analysis/figures.py" ;;
esac; done
