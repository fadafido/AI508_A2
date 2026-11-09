# Reproducing Garralda-Barrio et al. (FGCS 2025)

AI508 Big Data Technologies and Applications, Assignment 2. Fadi Alazayem (25002207), BUiD, November 2026.

The paper: M. Garralda-Barrio, C. Eiras-Franco and V. Bolón-Canedo, "Adaptive incremental transfer
learning for efficient performance modeling of big data workloads", Future Generation Computer Systems
166 (2025) 107730, https://doi.org/10.1016/j.future.2025.107730

It predicts how long a Spark workload will take under a given configuration. Each run is described by
a descriptor built from the task metrics in its Spark event log; for a new workload the model picks
similar past runs and fits a non-negative linear model on their configurations.

## What this repository does

1. Rebuilds the paper's input data from the raw Spark event logs the authors published: 4.1 GB, 215 runs
   of 24 HiBench workloads, 1.25 million tasks.
2. Rebuilds the descriptor extraction, which isn't published, once in plain Python and once as a PySpark
   job. The two agree to 1e-9, and the result is checked against the descriptor paper (JPDC 2024).
3. Runs the authors' model and the two baselines (Dou et al., Prats et al.) with the authors' code
   (a few documented fixes), LOOCV and LOGOCV on DS, DS' and DS'', and compares with Tables 5 to 12.
4. Re-runs the authors' later FGCS 2026 offline experiment unchanged, as a check of the model code.
5. Measures how the PySpark featuriser scales with the number of cores.
6. Re-runs 8 HiBench workloads on a Spark 2.4.8 + HDFS cluster I set up, and checks whether their
   descriptors still point to the right workload.

The main numbers are in `docs/results.md`, the full log of what I did in `WORKLOG.md`.

## Layout

```
src/data/          prepare_raw.py (join + extract the zip), audit.py (data vs papers)
src/featuriser/    parse_logs.py, build_descriptors.py (plain Python), descriptor.py (the maths),
                   spark_featuriser.py (PySpark), crosscheck.py, merge_spark_chunks.py
src/validation/    jpdc_clustering.py (featuriser vs JPDC 2024 Tables 4-6)
src/experiments/   datasets.py (DS, DS', DS''), cv.py (parallel CV around the authors' code),
                   run_fgcs2025.py, run_fgcs2026_anchor.py
src/analysis/      compare_with_paper.py, audit_paper_tables.py, cluster_rerun.py, scaling_summary.py,
                   figures.py
scripts/           run_all.sh, run_experiments.sh, scaling.sh, spark_featurise_chunked.sh,
                   cluster/ (HDFS + Spark standalone + HiBench)
patches/           fixes to the authors' code (apply_patches.py, authors_code.diff)
env/               requirements, environment set-up and capture
docs/              decisions.md, issues.md, environment.md, cluster.md, results.md
results/           paper_reference/ (published numbers), reproduced/ (our runs), comparison/, figures/
data/cluster/      event logs and run records of the re-runs
```

The raw event logs, intermediate files, the authors' repositories and the cluster software are not in
git. The scripts fetch or rebuild them.

## Running it

Linux or macOS, Python 3.10 or 3.11, Java 11 or 17 for PySpark.

```
bash scripts/fetch_external.sh        # the authors' repositories at the commits I used
bash env/setup_env.sh                 # virtual environments
bash scripts/run_all.sh               # everything except the cluster re-runs
```

For the cluster re-runs see `docs/cluster.md` (`scripts/cluster/setup_tools.sh`, then
`scripts/cluster/step.sh` until it says "all done").

## Checking the work

Three ways, from cheapest to most thorough:

1. Read the results. `docs/results.md` summarises every number and names the file under `results/`
   it comes from; `WORKLOG.md` says when and on which machine each one was produced.
2. A two-minute check, needs only the repository and the environments:
   ```
   ~/.venvs/ai508a2/bin/python -m pytest tests/      # descriptor maths against hand-computed values
   STEPS=patches bash scripts/run_all.sh             # rebuild the patched copy of the authors' code
   ~/.venvs/ai508a2/bin/python src/experiments/run_fgcs2025.py --check
   ```
   The last command runs the authors' own validation code and our parallel driver on DS and
   compares them; the descriptors it needs are in git (`data/processed/`).
3. The full rerun: `bash scripts/fetch_external.sh`, then
   `FORCE=--force bash scripts/run_all.sh` (hours; the raw event logs are joined from the authors'
   repository). The cluster re-runs are separate, see `docs/cluster.md`.

## Credits

Model and baseline code: github.com/mgarralda/garralda-performance-model (MIT licence), which also hosts
the event logs. FGCS 2026 framework and the authors' descriptor file:
github.com/mgarralda/spark-self-tuning-framework (CC BY-NC 4.0). HiBench (Apache 2.0), Apache Spark and
Hadoop (Apache 2.0). The method details and published tables were taken from the first author's open
access PhD thesis (hdl.handle.net/2183/48114).
