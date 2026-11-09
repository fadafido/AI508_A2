#!/usr/bin/env bash
# Model experiments of the FGCS 2025 reproduction. Usage: scripts/run_experiments.sh [stage ...]
# Stages: check core regressors lambda kmax variants baselines_ds baselines_dsp baselines_dspp
#         baselines_legacy_grid
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

stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=(check core regressors lambda kmax variants baselines_ds baselines_dsp baselines_dspp baselines_legacy_grid)
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
  # I-09 sensitivity: the grid with [1.0, 'sqrt'], 1.0 being what 'auto' meant for regressors in old
  # scikit-learn, next to the patched ['sqrt'] grid. Dou on DS and DS'' (where its grid is degenerate).
  baselines_legacy_grid)
    # AI508_LEGACY can point somewhere off the project mount (sed/rm misbehave on shared mounts)
    LEG=${AI508_LEGACY:-build/authors_code_legacy}
    rm -rf "$LEG" && mkdir -p "$LEG" && cp -R build/authors_code/. "$LEG/"
    for b in dou prats; do
      f="$LEG/garralda_performance_model/baselines/${b}_perf_model.py"
      $PY - "$f" <<'PYEOF'
import sys
p = sys.argv[1]
t = open(p).read()
old = "'rf__max_features': ['sqrt']"
assert old in t, p
t = t.replace(old + "  # [patch I-09] 'auto' no longer exists in scikit-learn 1.3+",
              "'rf__max_features': [1.0, 'sqrt']  # legacy grid (I-09 sensitivity)")
open(p, "w").write(t)
PYEOF
    done
    run base_legacy env AI508_AUTHORS="$(cd "$LEG" && pwd)" $R --datasets DS,DSpp --models dou --seeds 0-2 --tag-suffix _gridlegacy ;;
  *) echo "unknown stage $s"; exit 1 ;;
esac; done
