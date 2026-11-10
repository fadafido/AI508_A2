# Results

All numbers come from the files under `results/`, and all of them were recomputed on VM 2 in one run
(WORKLOG, 9 Nov). "Paper" means FGCS 2025 (or JPDC 2024 where said); "ours" is this repository. Times
are in seconds.

How the metrics are computed (as the code computes them, which is what we reproduce):

- LOOCV metrics are pooled over all samples. LOGOCV metrics are computed per held-out workload group
  and then averaged over groups (a macro-average), so small groups weigh as much as large ones.
- HME is the harmonic mean of MAE and RMSE (FGCS Eq. 13); AME is the arithmetic mean of the LOOCV HME
  and the LOGOCV HME (Eq. 14). Lower is better.
- The model's lambda weighs the two neighbour-quality criteria. The code computes
  lambda*CqC + (1-lambda)*CdV, which is the paper's Eq. 7 with the roles of lambda swapped (I-11); all
  lambda values here use the code's convention.
- The base learner the paper calls NNLS is non-negative least squares with a free intercept: a
  Yeo-Johnson transform of the target around a pipeline of min-max scaling and
  `LinearRegression(positive=True)` (I-13). Only the coefficients are non-negative; predictions can be
  negative, and the inverse transform can amplify extrapolations (see the svm fold below).
- Aggregation over seeds: every metric is computed per seed, AME included, and the reported value is
  the median over seeds, for every model and dataset alike. The median is used because a few DS'' seeds
  blow up in LOGOCV (I-18) and would dominate any mean; the mean, min, max and sd are in
  `results/comparison/fgcs2025_vs_ours.csv` (sd is NaN where only one seed exists, i.e. for the
  deterministic Garralda model on DS). Baselines ran seeds 0-2 everywhere, the Garralda model 10 seeds
  on DS' and DS'' (it is deterministic on DS), so `results/comparison/matched_seeds.csv` holds all
  three models on the same seeds 0-2 and one machine.
- Coverage (finite predictions / all samples) is in the comparison table per CV. It is exactly 1 for
  both baselines everywhere; the Garralda model drops one prediction (a Yeo-Johnson fit failure, I-14)
  in 4 of 10 DS' seeds and in seeds 0 and 4 on DS'', so its worst coverage is 429/430.

## 1. Data and featuriser

- 215 of the paper's 216 runs are in the published logs (I-01): 1,251,335 tasks, 4.09 GB.
- The PySpark featuriser and the plain Python one agree to 3.5e-10 on all 21,500 descriptor values, with
  identical task counts and input sizes (`results/reproduced/featuriser_crosscheck.json`).
- Validation against the descriptor paper (JPDC 2024, `results/reproduced/jpdc_validation/`). One
  choice was calibrated on the JPDC tables: the task order inside a series (Task ID order fits Table 5
  better than log order, D-02). Everything else was fixed from the authors' own descriptor file before
  the comparison (D-03 to D-08), so the following are independent checks:
  - Table 4 (clustering quality, 12 configurations): mean absolute difference 0.032 in silhouette,
    0.038 in F-measure, 0.030 in NMI. Degree 3 is the worst option in both. The paper's best is Cusum
    with degree 2 (NMI 0.902); ours is Cusum with degree 1 (0.913), with degree 2 close behind (0.907).
  - Table 5 (mean distance per workload): median absolute difference 0.14. Dataset diameter 28.19
    against 27.55.
  - Table 6: the Garralda descriptor beats the Prats descriptor on F-measure and NMI in all 8
    size/algorithm cells, as in the paper.
- Scaling of the PySpark featuriser on VM 2 (4.09 GB, 9 jobs per run, median of 2 repetitions, every
  run reproduces the reference descriptors; `results/reproduced/scaling/summary.csv`, per repetition
  and per phase in `runs.csv`). Two sets: the logs on the VM's own disk, and the same logs read
  through the shared project mount (the I/O-bound case):

  | Logs on | Cores | Total (s) | Session start (s) | Processing (s) | Speed-up (processing) | MB/s |
  |---|---|---|---|---|---|---|
  | VM disk | 1 | 198 | 20 | 179 | 1.00 | 20.6 |
  | VM disk | 2 | 147 | 18 | 129 | 1.39 | 27.8 |
  | VM disk | 4 | 125 | 18 | 107 | 1.66 | 32.6 |
  | VM disk | 8 | 138 | 21 | 117 | 1.53 | 29.7 |
  | shared mount | 1 | 265 | 22 | 243 | 1.00 | 15.4 |
  | shared mount | 2 | 213 | 22 | 191 | 1.27 | 19.2 |
  | shared mount | 4 | 184 | 22 | 162 | 1.50 | 22.3 |
  | shared mount | 8 | 206 | 23 | 182 | 1.33 | 19.9 |

  The Spark session start is about 20 s and constant, so it is not the main limit. Storage matters:
  from the VM disk every core count is 25 to 33% faster than through the mount, and the repetitions
  are far more repeatable (the two 1-core runs differ by 0.6 s on the disk and by 52 s through the
  mount). But the shape survives the move: even from the local disk the processing part speeds up
  only 1.7x at 4 cores and gets slower again at 8, so beyond 4 cores the limit is not the storage
  either; the work is split into 9 separate Spark jobs (one per setting folder) whose few large files
  bound the useful parallelism at this data size. VM 1 measured 206, 154, 141 s at 1, 2, 4 cores on
  its faster mount, with the same shape.

## 2. Check of the model code (FGCS 2026 anchor)

The authors' later offline experiment (800 runs with their own descriptors), run unchanged, gives
exactly the published file to 3 decimals: MAE 163.045, RMSE 517.132, P95 903.808, max 4431.059,
Kendall tau 0.694, precision@10 0.805 (`results/reproduced/fgcs2026_anchor/`). This is a positive
control for the model code and my environment: it is the authors' later code on their own descriptors,
in its own environment, so it shows the model implementation and my machine are not the problem. It is
not evidence about the 2025 pipeline (data preparation and descriptor extraction), which is where the
differences below must come from.

## 3. FGCS 2025

Values are the median over seeds (see the rule above). DS' and DS'' are synthetic copies of DS (noise
on the time; scaled descriptors and renamed types), not new workloads, so they mainly test robustness
to perturbations, not generalisation. Note that in LOOCV and LOGOCV alike, the prediction for a run is
conditional on that run's own descriptor: the query descriptor comes from the run being predicted,
which is the paper's protocol too.

**DS (Table 6)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 136.4 / 107.3 | 293.6 / 220.3 | 273.0 / 215.2 |
| Dou | 229.0 / 205.4 | 234.5 / 228.5 | 319.5 / 298.8 |
| Garralda | 88.5 / 117.0 | 225.4 / 242.5 | 199.9 / 245.3 |

**DS' (Table 8)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 44.1 / 35.9 | 254.8 / 222.7 | 183.4 / 160.2 |
| Dou | 159.9 / 157.6 | 241.4 / 257.3 | 272.8 / 283.8 |
| Garralda | 52.7 / 44.5 | 219.9 / 232.8 | 174.8 / 175.6 |

**DS'' (Table 10)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 61.1 / 69.7 | 160.6 / 131.5 | 143.5 / 133.7 |
| Dou | 168.8 / 88.1 | 199.8 / 176.5 | 265.7 / 181.5 |
| Garralda (10 seeds) | 59.8 / 44.7 | 108.6 / 108.4 | 111.6 / 100.1 |

Garralda on DS'': in 4 of the 10 seeds a few LOGOCV predictions explode (I-18), which is why the
LOGOCV MAE has sd 322,772 across seeds while its median sits on the paper. The other 6 seeds are close
to the paper on their own.

**The DS gap, per model.** Our Garralda AME on DS is 23% above the paper, while both baselines land
below their published values, and the three cases need different explanations:

- The published logs are probably not the runs behind the paper's tables: run-time statistics match
  FGCS Table 3 within 1.5 s for only 15 of the 24 workloads, 8 differ a lot, and 1 (terasort) differs
  by up to 5% (I-07). Rescoring our DS errors by these audit classes
  (`results/comparison/ds_gap.json`): the Garralda model's LOGOCV MAE is 340 s on the "different runs"
  workloads against 202 s on the consistent ones, so the data mismatch hits the proposed model hardest
  exactly where the inputs differ. For Dou the pattern is reversed (its errors concentrate on the
  consistent, long-running workloads), so data mismatch cannot explain its improvement; its grid is
  simply version- and seed-sensitive (I-09, I-10).
- The Garralda model is deterministic on DS, so random-forest arguments (I-10) cannot explain its gap;
  the candidates are the input data (I-07) and the rebuilt descriptor extraction (D-02 to D-08). Our
  own audit counts svm among the consistent workloads, so the svm fold below is not explained by I-07.
- Prats: both public revisions of the authors' baseline file (283bc1a and f2d3f93) contain the same
  broken LOGOCV branch (an indentation error, a call with keyword arguments the model builder does not
  accept, and the whole held-out group predicted as one 3-D input, I-05/I-06), so the code that
  produced the paper's Prats numbers is not public, and we cannot know what it did. Our patched branch
  (D-16) gives Prats LOGOCV MAE 220 against the published 294, which is most of why Prats is stronger
  in our runs.

**One svm fold dominates the Garralda LOOCV RMSE on DS** (`results/comparison/ds_gap.json`): the run
with true time 4,819 s is predicted as 11,469 s. Its k = 8 neighbours are all svm runs with times from
25 to 6,216 s; the NNLS prediction in the Yeo-Johnson space is 1.643, just above the largest
transformed neighbour target (1.560), and the inverse transform turns that small extrapolation into
11,469 s. This one fold carries 82.5% of the total LOOCV squared error; without it the RMSE drops from
499 to 209.

**Dou's grid search is degenerate on DS''** (I-09): with k = 5 neighbours and 5-fold CV every candidate
scores NaN (0 of 15 finite; on DS all 15 are finite), so the refit silently takes the first grid point
(max_depth 10, max_features 'sqrt', 10 trees) and the "grid search" is one fixed forest. A sensitivity
run with the legacy grid (max_features [1.0, 'sqrt']; 1.0 is what 'auto' meant for regressors in old
scikit-learn) is in `results/reproduced/fgcs2025/*/dou_gridlegacy/` (seeds 0-2): on DS it gives Dou
AME 307.7 against 298.8 with the patched grid, closer to the published 319.5, which fits the authors
running on an older scikit-learn where the full 'auto' half of the grid still worked. On DS'' the
search is degenerate under both grids and only the silent fallback forest changes (max_features 1.0
instead of 'sqrt'): the median AME barely moves (182.8 against 181.5) but the seed spread widens
(150.6 to 208.5 against 154.3 to 192.8), so Dou's DS'' numbers are an artefact of which fixed forest
the fallback lands on, under the paper's grid as much as under ours.

**Base learner ablation (Table 5, DS, median of 3 seeds for the random learners)**

| Learner | LOOCV MAE paper / ours | LOGOCV MAE paper / ours |
|---|---|---|
| Random forest | 115.3 / 110.3 | 257.8 / 270.3 |
| AdaBoost | 123.3 / 113.6 | 293.5 / 270.8 |
| Gradient boosting | 115.6 / 106.8 | 269.6 / 264.9 |
| NNLS (the paper's choice) | 88.5 / 117.0 | 225.4 / 242.5 |

NNLS has the lowest LOGOCV error in both, but our LOOCV error is the highest of the four: the svm fold
above dominates it.

**Headline claim.** The paper says its model lowers AME by 18% against Prats and 45% against Dou (31%
overall); recomputed from its own Table 12 that is 17.9% and 43.8%. Ours, with the median-over-seeds
rule: 33.6% against Dou (17.9% on DS, 38.1% on DS', 44.8% on DS'') and 0.5% against Prats (-14.0% on
DS, -9.7% on DS', +25.1% on DS''). So the model clearly beats Dou, as in the paper, but it does not
beat Prats in our runs: the patched Prats baseline is much stronger than its published numbers, and the
code behind those numbers is not public (see above).

**Sensitivity (DS, AME)**

- lambda (code convention, I-11): 275.7 at 0, 245.3 at 0.5 (default), 193.2 at 1.0. With lambda near 1
  our AME drops below the paper's 199.9.
- Upper bound of k: 204.6 at 6, 245.3 at 18 (default), 285.6 at 30.
- Descriptor variants: log order 275.0, no Cusum 190.0, Cusum with degree 1 182.1, Cusum with degree 3
  293.0 (default 245.3). The simpler descriptors do better on DS.

**Same code on two CPU types (I-19).** The cloud VM's x86-64 results for the DS'/DS'' baselines (seed
0) are kept in `results/crosscheck_x86/`. On DS the Garralda model and Dou give the same numbers on
ARM64 and x86-64 at the precision reported here; DS' and DS'' differ in the second or third significant
digit, and Prats on DS has AME 215.2 on ARM64 and 220.0 on x86-64. The two ARM64 VMs (VM 1 and VM 2)
agree exactly on every value checked.

## 4. Re-runs on my own cluster

Two campaigns of HiBench re-runs (`docs/cluster.md`): campaign 1 (VM 1, 69 runs, executor memory / 4)
and campaign 2 (VM 2, 72 runs including bayes small, memory / 2). The analysis treats each re-run as a
query against the gallery of 215 runs from 2022: both are put in the same min-max-scaled Garralda
space (the scaler fitted on the gallery, as in the model's kNN) and the question is which gallery run
is nearest. The leave-one-out reference row answers the same question for the gallery itself, with the
scaler refitted without the held-out run.

| | Campaign 1 (Garralda / Prats) | Campaign 2 (Garralda / Prats) |
|---|---|---|
| Nearest 2022 run is the same workload | 95.7% / 91.3% (66, 63 of 69) | 98.6% / 94.4% (71, 68 of 72) |
| ... among the 3 nearest | 97.1% / 92.8% | 100% / 94.4% |
| Leave-one-out inside 2022, same 8 workloads | 100% / 83.3% | same gallery |
| Nearest run has the same HiBench size | 56.5% / 97.1% | 63.9% / 97.2% |

Campaign 1's misses: 2 bayes runs (nearest gbt and pca) and 1 sleep run (nearest scan) for Garralda;
Prats mistakes wordcount for pagerank 6 times. Campaign 2: 1 repartition run (nearest terasort) for
Garralda, the same wordcount/pagerank confusion for Prats. The size row is a real trade-off: the
Garralda descriptor identifies the workload much better than Prats (and 100% against 83.3% on the
2022 gallery itself) but the workload size much worse (56-64% against 97%; on the full 2022 gallery
59.5% against 83.7%), because its shape features are nearly size-invariant by construction.

Distances in the scaled Garralda space (campaign 1, medians): a re-run is 1.38 from its 2022 twin
(same workload, size and setting), about the same as two 2022 runs of one workload under different
settings (1.42), and far from other workloads (3.42). Two repetitions of the same job on the VM are
0.99 apart. Campaign 2 gives the same picture (1.39, 1.44, 3.39, 0.99), and matched runs of the two
campaigns are 1.12 apart, i.e. a resource change moves the descriptor about as much as run-to-run
noise (`docs/cluster.md`). The features that move most between platforms are timing ones (result
serialisation, fetch wait, deserialisation), which also vary most between repetitions.

Run times on the VMs are much shorter than in 2022 (median ratio 0.37 on VM 1, 0.30 on VM 2). This
ratio is not a speed-up claim: the VMs are one machine with faster cores, no YARN container start-up
and no network, and executor memory was cut, so only the relative order is meaningful. That order is
kept fairly well (Spearman 0.72 over 45 cells on VM 1, 0.70 over 48 on VM 2).

Feeding the re-runs to the model as queries (campaign 1; `cluster_rerun*/model_predictions.csv`), with
the paper's nominal configuration vector and, as a variant, the settings the VM really ran with
(mainly a smaller executor memory):

| History used for the prediction | MAE nominal / actual (s) | MAPE nominal / actual |
|---|---|---|
| 2022 runs | 19.8 / 16.3 | 271% / 218% |
| 2022 runs + VM runs of the other workloads | 14.7 / 11.2 | 194% / 133% |
| VM runs of the other workloads only | 6.6 / 6.7 | 59% / 59% |

Campaign 2 behaves the same (330% / 318%, 248% / 192%, 50% / 50%; for the VM-only history the two
variants coincide because the actual settings are the nominal ones with the memory halved, which the
model's min-max scaling absorbs). The descriptor transfers across platforms, but the run times don't:
the model only becomes useful on the new platform once it has runs from that platform in its history.
That is consistent with the paper's incremental story, but it is not the paper's incremental
adaptation experiment: moving to a new platform is a harder shift than the paper's synthetic DS' and
DS'' perturbations, and the model here was used as-is, without the paper's adaptation mechanism.
