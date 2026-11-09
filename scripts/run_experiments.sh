#!/usr/bin/env bash
# Model experiments of the FGCS 2025 reproduction. Usage: scripts/run_experiments.sh [stage ...]
# Stages: check core regressors lambda kmax variants baselines_ds baselines_dsp baselines_dspp
# Every stage skips results that already exist (delete the JSON or pass FORCE=--force to recompute).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
JOBS=${JOBS:--1}
FORCE=${FORCE:-}
R="$PY src/experiments/run_fgcs2025.py --jobs $JOBS $FORCE"
mkdir -p logs
stamp() { date +%Y%m%d_%H%M%S; }
run() { local name=$1; shift; echo "[$(date '+%F %T')] start $name"; "$@" 2>&1 | tee -a "logs/experiments_${name}_$(stamp).log"; echo "[$(date '+%F %T')] end $name"; }

stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=(check core regressors lambda kmax variants baselines_ds baselines_dsp baselines_dspp)
for s in "${stages[@]}"; do case $s in
  check)        run check $R --check ;;
  # Garralda model, NNLS, lambda 0.5: DS is deterministic; DS'/DS'' depend on the synthetic noise -> 10 seeds
  core)         run core_ds $R --datasets DS --models garralda --seeds 0
                run core_dsp $R --datasets DSp,DSpp --models garralda --seeds 0-9 ;;
  # FGCS Table 5: base-learner ablation (random learners -> 3 seeds)
  regressors)   run regressors $R --datasets DS --models garralda --regressors rf,adaboost,gb --seeds 0-2 ;;
  # FGCS Appendix B: lambda sweep on DS
  lambda)       run lambda $R --datasets DS --models garralda --lambdas 0,0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9,1 --seeds 0 ;;
  # sensitivity to the neighbourhood upper bound (RoT gives 18 on DS)
  kmax)         for k in 6 10 14 24 30; do run kmax$k $R --datasets DS --models garralda --kmax $k --seeds 0; done ;;
  # featuriser ablation propagated to the performance model
  variants)     for v in order_log no_cusum_d2 cusum_d1 cusum_d3; do run var_$v $R --datasets DS --models garralda --variant $v --seeds 0; done ;;
  # baselines (random forests with grid search: the expensive part); seeds 0-2 so all three models
  # share the same seeds on every dataset
  baselines_ds)   run base_ds $R --datasets DS --models dou,prats --seeds 0-2 ;;
  baselines_dsp)  run base_dsp $R --datasets DSp --models dou,prats --seeds 0-2 ;;
  baselines_dspp) run base_dspp $R --datasets DSpp --models dou,prats --seeds 0-2 ;;
  *) echo "unknown stage $s"; exit 1 ;;
esac; done
