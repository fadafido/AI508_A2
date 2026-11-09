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
