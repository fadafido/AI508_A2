# Work log

Fadi Alazayem (25002207), AI508 Assignment 2. What I did, in order, and where each result is. Times are
UTC. "Mac" is the Linux VM on my MacBook Pro (ARM64, 4 vCPUs); "cloud VM" is an x86-64 Linux machine
with 2 vCPUs that I used to develop the pipeline and for the long baseline runs. Decisions are D-xx in
`docs/decisions.md`, problems are I-xx in `docs/issues.md`.

## 9 November 2026

### Choosing the paper
- Went through the candidate papers from my Assignment 1 corpus, cloned every repository and ran what
  could run offline. Picked Garralda-Barrio et al., FGCS 166 (2025) 107730: the raw event logs and the
  model code are public, and the setting (Spark on a cluster) is a real Big Data system.
- The first author's PhD thesis is open access (hdl.handle.net/2183/48114) and reprints the JPDC 2024,
  FGCS 2025 and FGCS 2026 papers in full. The method details and published tables I used come from
  there.

### Data, featuriser and validation (cloud VM)
- 10:55 Project skeleton. The authors' repositories copied unmodified into `external/` at fixed commits.
- 10:58 Joined the 7-part zip from the authors' repository: 4.09 GB, 215 Spark event logs, CRC ok
  (`data/raw/MANIFEST.json`).
- 11:00 Plain Python parser over all logs: 215 applications, 1,251,335 task events with metrics, 3 failed
  tasks, 23 s on 2 cores. Spark 2.4.8 keeps peak execution memory only in the task accumulables (D-07).
- 11:02 Worked out the parts of the descriptor the papers don't describe from the authors' own
  descriptor file: min-max scaling after the Cusum, x = linspace(0, 1, n), numpy.polyfit, tasks per
  stage as one value per stage. For 2-stage applications their file holds exactly (-0.5, -0.5, 1, 0) and
  (0.5, 0.5, -0.0, 0), which is what this construction gives (D-03, D-04, D-06).
- 11:05 Data audit (`results/reproduced/data_audit/`): one run missing (I-01), a folder name that doesn't
  match its runs (I-02), and run times that match FGCS Table 3 for only 15 of the 24 workloads, so the
  published logs aren't exactly the runs behind the tables (I-07). Table 3's aggregate std doesn't add
  up (I-08).
- 11:08 Checked the featuriser against JPDC 2024 (`results/reproduced/jpdc_validation/`). k-means with
  Cusum and degree 2: F-measure 0.851 (paper 0.846), NMI 0.907 (0.902), silhouette 0.346 (0.321).
  Diameter 28.19 (27.55). Garralda beats Prats on F and NMI in all 8 cells, like in the paper. Task ID
  order fits Table 5 better than log order, so I used it (D-02).
- 11:12 Four fixes were needed before the authors' code would run (I-03 to I-06,
  `patches/authors_code.diff`).
- 11:13 My parallel CV driver gives the same numbers as the authors' `validation.py` to 1e-9
  (`results/reproduced/fgcs2025/driver_equivalence.json`).
- 11:14 PySpark featuriser (window functions and moment sums, nothing collected per application): agrees
  with the plain version to 3.5e-10 on all 21,500 descriptor values.
- 11:15 to 12:00 Model experiments on the cloud VM. Proposed model on DS: LOOCV MAE 116.97, LOGOCV MAE
  242.48, AME 245.28 (paper 88.47, 225.38, 199.85). One svm run (11,469 s predicted for 4,819 s) dominates
  the LOOCV RMSE. Baselines on DS: Dou AME 295.03, Prats 219.96 (paper 319.51 and 273.02).

### Moving to the Mac
- 11:40 to 11:50 Same pipeline on the Mac. The zip has the same checksum, parsing takes 28 s, audit and
  JPDC validation give the same numbers. The VM stops long commands after 3 minutes, so the PySpark
  featuriser ran one setting folder at a time (9 jobs, 141 s in total) and the parts were merged; the
  cross-check gives the same 3.5e-10.
- 11:50 Model experiments on the Mac: driver check identical, DS identical to the cloud VM. DS' and DS''
  agree with the cloud VM to 2 or 3 significant digits but not bit for bit (I-19).
- 11:52 FGCS 2026 anchor (the authors' later offline experiment, 800 executions, their descriptors):
  identical to the published results file to 3 decimals, on the Mac too.
- 11:55 Base learner ablation, lambda sweep, k_max sweep and descriptor variants on the Mac (about a
  minute in total).

### Cluster re-runs (Mac)
- 11:58 Tested the cluster scripts on the cloud VM first: HDFS, Spark 2.4.8 standalone and a HiBench sort
  run work and produce a normal event log.
- 12:09 First start on the Mac failed: the VM's hostname doesn't resolve and the NameNode won't start.
  Running the session in a private UTS namespace with hostname localhost fixed it.
- 12:14 Disk full after about 15 runs: every executor gets its own copy of the 118 MB HiBench jar.
  Now deleted after each run.
- 12:20 ScalaSleep had 2 tasks instead of one per core. Standalone Spark doesn't wait for executors
  before the first job; YARN does. Set minRegisteredResourcesRatio=0.8 and re-ran the tiny jobs done so
  far (old ones kept in `data/cluster/superseded/`).
- 12:22 to 12:50 All 48 tiny-size runs (8 workloads x 3 settings x 2 repetitions).
- 12:36 to 13:04 Small-size runs. kmeans (settings 2 and 3) and repartition (setting 2) had executors
  killed by the OOM killer, re-ran fine with one worker daemon and 512 MB executors. bayes failed every
  time, even with 1 GB for the driver, so it's left out. 69 runs in the end (`docs/cluster.md`).

### Analysis
- 13:10 Featurised the 69 re-runs and compared them with the 2022 runs
  (`results/reproduced/cluster_rerun/`). The nearest 2022 run is the same workload for 66 of 69 re-runs
  with the Garralda descriptor (95.7%) and 63 of 69 with the Prats descriptor (91.3%). A re-run is about
  as far from its 2022 twin as two 2022 runs of the same workload under different settings, and much
  further from other workloads. Run times are shorter on the VM (median ratio 0.37) but keep their order
  (Spearman 0.72). Using the 2022 runs as history, the model's predictions for the VM runs are off by
  271% on average; with only VM runs of other workloads as history the error drops to 59%.
- 13:17 DS baselines on the Mac: Dou AME 295.03, the same as on the cloud VM. Prats 215.20 against 219.96
  on the cloud VM; the Prats descriptors are identical, so the difference comes from the random forests
  on the two CPU types (I-19).
- 13:19 to 13:45 Featuriser scaling on the Mac, two repetitions each: local[1] 206 s, local[2] 154 s,
  local[4] 141 s for the full 4.09 GB (speed-up 1.34 and 1.47). With 9 separate jobs, start-up costs
  take a big share at this data size.
- 13:28 Dou crashed on DS'' on the cloud VM. With k = 5 every grid-search score is NaN and the refit lands
  on max_features='auto', which scikit-learn 1.4 rejects (I-09). Removed 'auto' from the grid; DS
  predictions are unchanged, checked unit by unit. Restarted the DS'' baselines and the extra DS seeds.
- 13:35 Comparison script (`src/analysis/compare_with_paper.py`) and figures (`src/analysis/figures.py`).
- 13:50 Mac went offline in the middle of the extra DS baseline seeds.
- 14:00 to 14:20 Extra DS baseline seeds on the Mac: Dou AME 298.75 and 303.18, Prats 219.51 and 211.14
  (seed 0: 295.03 and 215.20).
- 14:45 Prats on DS'' finished on the cloud VM: AME 135.35 (paper 143.45). Final comparison tables and
  figures, summary in `docs/results.md`. Against the paper's headline: the model beats Dou by 35% on
  average (paper 45%) but is only 1% better than Prats (paper 18%).

### VM 2 (second VM on the Mac)
- 16:45 VM 1 kills every session after 3 minutes, so the remaining work moves to a second Linux VM on
  the same Mac, built with Multipass: Ubuntu 22.04.5, ARM64, 8 vCPUs, 10 GB RAM, 40 GB disk ("VM 2").
  The project folder is mounted into the VM; the venvs (~/.venvs) and the cluster work folder
  (~/ai508-cluster) stay on the VM's own disk because HDFS and Spark fail on the shared mount when
  they chmod their directories.
- 17:10 Same tool chain as VM 1: venvs with the pinned package versions (checked), JDK 8u512 aarch64,
  Spark 2.4.8 and Hadoop 2.7.7; all downloads match `tools/downloads.sha256` and the HiBench jars
  match `tools/hibench/SHA256SUMS`. Environment capture: `results/env/mac_m3pro_multipass_vm.txt`.
- 17:14 The cluster keeps binding to 127.0.0.1; a socat systemd service forwards the web UIs to the
  VM's address so I can watch runs from the Mac browser (8080 Spark master, 4040 running job, 50070
  HDFS, 18080 a Spark history server over the event-log folder). Start test: cluster up in 21 s, one
  live datanode, 3 workers, all three UIs answer from the Mac. The VM's hostname resolves, so the
  unshare workaround of VM 1 doesn't trigger.

### Cluster campaign 2 (VM 2)
- 17:16 Campaign plumbing: env.sh takes the output folder (AI508_OUT) and a campaign id (AI508_CAMPAIGN,
  the Spark benchmark group id, also written into runs.jsonl), run_all.sh and cluster_rerun.py take the
  campaign folder. Campaign 2: data/cluster2/, group id rerun-vm2, executor memory / 2 instead of / 4
  (D-20). Same 86-job plan as campaign 1 (seed 7) plus bayes small, which VM 1 couldn't run. Smoke test
  (prepare + sort tiny under setting 1), then the whole plan in tmux.
- 17:19 to 17:32 First pass: 73 of 86 jobs ok in 777 s. The pagerank and bayes data generation failed:
  the generator shares its dictionary and Zipf tables through DistributedCache symlinks created in the
  working directory, and the Multipass project mount refuses symlinks (I-20). The 9 pagerank runs read
  a 4-byte leftover input and finished in 3 s as "ok"; moved to data/cluster2/superseded/. hibench.sh
  now changes into the work folder on the VM disk first, and step.sh skips runs whose data generation
  failed in the same session.
- 17:38 to 17:43 Re-ran the 22 open jobs: all 86 ok (881 s of job time in total). bayes small works on
  VM 2 (3 GB of executors against 10 GB of RAM), so campaign 2 has all 72 runs, 9 per workload.
- 17:45 Featurised and analysed campaign 2 (results/reproduced/cluster_rerun2/). Against 2022: top-1
  recognition 71/72 (98.6%) for the Garralda descriptor (one repartition picks terasort), 68/72 (94.4%)
  for Prats (wordcount picked as pagerank 4 times, like campaign 1). Size identification stays the
  trade-off: 63.9% (Garralda) against 97.2% (Prats). Run times: median ratio 0.304 to 2022, Spearman
  0.70 over 48 cells. Model on the re-runs (nominal settings): MAPE 330% with 2022 history, 248%
  adding VM runs of other workloads, 50% with VM runs only; with the VM's actual executor memory in
  the query vector: 318% and 192% (identical for VM-only history: the actual settings are the nominal
  ones with memory halved, which the model's min-max scaling absorbs).
- 17:46 Campaign 2 against campaign 1 (results/reproduced/cluster_rerun2/vs_campaign1.json): matched
  runs are 1.12 apart in the scaled descriptor space, about the rep 1 vs rep 2 noise inside one
  campaign (0.99 and 0.99) and far from other workloads (3.36); every campaign-2 run's nearest
  campaign-1 run is the same workload (69/69). Campaign 2 runs 21% faster (median ratio 0.79,
  Spearman 0.97). The descriptor stays put when only the resources change; what moves are the timing
  features (serialisation, fetch wait), the same ones that move between the VM and 2022.

### The whole reproduction again (VM 2)
- 17:44 The four DS' and DS'' baseline results computed on the x86-64 cloud VM are copied to
  results/crosscheck_x86/ so the ARM64 vs x86-64 comparison (I-19) keeps its evidence. The baselines
  now run seeds 0-2 on DS, DS' and DS'' by default (run_experiments.sh), so all three models share
  the same seeds everywhere. compare_with_paper.py now computes every metric per seed and aggregates
  with one rule (the median over seeds, because the DS'' blow-up seeds of I-18 would dominate any
  mean), marks sd as not estimated with one seed, adds coverage columns (finite predictions / all)
  and writes a matched-seed table (results/comparison/matched_seeds.csv).
- 17:48 to 20:21 Everything fresh on VM 2, one command: data/interim and the chunked featuriser
  outputs deleted first, then run_all.sh with FORCE over env, patches, raw, parse, descriptors,
  audit, validate, spark, paper_audit, anchor, experiments, scaling, reruns and compare. 2 h 33 min,
  almost all of it the six baseline stages. The deterministic results match VM 1 exactly: all DS
  values (Garralda AME 245.28; Dou 295.03/298.75/303.18; Prats 215.20/219.51/211.14 for seeds 0-2),
  the Garralda model on DS' and DS'' (mean 176.11, median 100.13), the FGCS 2026 anchor (163.045 /
  517.132, identical to the published file), the JPDC validation (diffs 0.032/0.038/0.030, diameter
  28.19), the featuriser cross-check (3.5e-10) and the campaign-1 recognition (66/69 and 63/69).
- 20:21 New numbers from this run: the DS' and DS'' baselines on ARM64 with seeds 0-2 (median AME:
  Prats 160.17 on DS', 133.71 on DS''; Dou 283.77 and 181.49) against the single x86-64 seed before
  (162.5, 135.4, 284.1, 192.8); the biggest mover is Dou on DS'', whose degenerate grid (I-09) makes
  it the most seed- and machine-sensitive result. With the single aggregation rule the headline is:
  the model beats Dou by 33.6% on average (paper's own Table 12 gives 43.8%) and Prats by 0.5%
  (paper 17.9%).
- 20:25 The scaling step had quietly reused the VM 1 chunk outputs (chunks are skipped when present),
  so its 1-, 2- and 4-core times were VM 1 times; deleted data/processed/scaling and re-ran all four
  core counts on an otherwise idle VM 2: 265, 213, 184, 206 s for 1, 2, 4, 8 cores (medians of 2,
  every run reproduces the reference descriptors). The Spark session start is 22 s and roughly
  constant; the processing part speeds up 1.27x, 1.50x, 1.33x. Reading the 4.1 GB through the shared
  mount makes VM 2 slower than VM 1 here, and 8 cores are no faster than 4: the featuriser is I/O
  bound at this data size, not start-up bound.

### Fixes from the external review
- 20:30 Known-answer tests for the descriptor maths (tests/test_descriptor.py, pytest: hand-computed
  min-max, z-score, Cusum, final min-max and degree-2 fit on tiny series, the authors'-file identities
  for 2-point series, scale invariance; 9 tests). A "Checking the work" section in the README: read
  the results, a two-minute check (tests + the driver-equivalence check), or the full rerun.
- 20:40 The DS gap pulled apart (src/analysis/ds_gap.py -> results/comparison/ds_gap.json). The worst
  LOOCV fold recomputed with the authors' two stages: svm, true 4,819 s, predicted 11,469 s (stored
  and recomputed agree), k = 8 all-svm neighbours with times 25 to 6,216 s; the prediction in the
  Yeo-Johnson space (1.643) lies just above the largest transformed neighbour target (1.560) and the
  inverse transform blows the overshoot up. That one fold carries 82.5% of the LOOCV squared error
  (RMSE 499 without it 209). DS errors rescored by the audit classes of I-07 (15 consistent, 8
  different runs, 1 terasort): the Garralda model's LOGOCV MAE is 340 s on the "different runs"
  workloads against 202 s on the consistent ones; for Dou the pattern is reversed.
- 20:46 Verified Dou's degenerate grid on DS'' directly: with k = 5 all 15 grid scores are NaN and the
  refit silently takes the first grid point (max_depth 10, max_features 'sqrt', 10 trees); on DS all
  15 are finite. Documented in I-09.
- 20:55 to 23:20 Legacy-grid sensitivity (new run_experiments.sh stage baselines_legacy_grid: a copy
  of the patched code with max_features [1.0, 'sqrt'], Dou, DS and DS'', seeds 0-2). DS: AME 307.7
  against 298.8 with the patched grid, closer to the published 319.5. DS'': the search is degenerate
  under both grids, the median barely moves (182.8 against 181.5) but the seed spread widens (150.6 to
  208.5 against 154.3 to 192.8). Also checked at the source that both public revisions of the authors'
  Prats file (283bc1a, f2d3f93) contain the same broken LOGOCV branch.
- 23:30 docs/results.md rewritten: metric definitions as the code computes them, the single
  aggregation rule and coverage, the DS gap per model, the svm fold, the Dou degeneracy with the
  legacy grid next to it, the corrected scaling explanation, both re-run campaigns described as
  queries against the 2022 gallery, the size-identification trade-off, and the ratio and incremental
  caveats. docs/cluster.md got the campaign 2 section, docs/environment.md the VM 2 table.

## 10 November 2026

- 02:50 to 03:20 Featuriser scaling repeated with the event logs copied to the VM's own disk
  (scaling.sh and spark_featurise_chunked.sh take LOGS and PREFIX now); the shared-mount set is kept
  next to it as the I/O-bound case. VM disk, medians of 2: 198, 147, 125, 138 s for 1, 2, 4, 8 cores;
  shared mount: 265, 213, 184, 206 s. Every run reproduces the reference descriptors. From the disk
  every core count is 25 to 33% faster and the repetitions are far more repeatable (0.6 s apart at
  1 core against 52 s through the mount), but the shape stays: the processing speed-up peaks at 1.7x
  on 4 cores and drops at 8, so beyond 4 cores the limit is the 9-job structure with its few large
  files per job, not the storage. Table, figure and explanation updated in docs/results.md.
