# Results

All numbers come from the files under `results/`. "Paper" means FGCS 2025 (or JPDC 2024 where said);
"ours" is this repository. Models with randomness were run with several seeds: the mean is given, and
for DS'' also the median because a few seeds blow up (I-18). Times are in seconds.

## 1. Data and featuriser

- 215 of the paper's 216 runs are in the published logs (I-01): 1,251,335 tasks, 4.09 GB.
- The PySpark featuriser and the plain Python one agree to 3.5e-10 on all 21,500 descriptor values, with
  identical task counts and input sizes (`results/reproduced/featuriser_crosscheck.json`).
- Against the descriptor paper (JPDC 2024, `results/reproduced/jpdc_validation/`):
  - Table 4 (clustering quality, 12 configurations): mean absolute difference 0.032 in silhouette, 0.038
    in F-measure, 0.030 in NMI. Degree 3 is the worst option in both. The paper's best is Cusum with
    degree 2 (NMI 0.902); ours is Cusum with degree 1 (0.913), with degree 2 close behind (0.907).
  - Table 5 (mean distance per workload): median absolute difference 0.14. Dataset diameter 28.19 against
    27.55.
  - Table 6: the Garralda descriptor beats the Prats descriptor on F-measure and NMI in all 8
    size/algorithm cells, as in the paper.
- Scaling of the PySpark featuriser on the Mac VM (4.09 GB, 9 jobs, median of 2 runs,
  `results/reproduced/scaling/summary.csv`):

  | Cores | Time (s) | Speed-up | MB/s |
  |---|---|---|---|
  | 1 | 206 | 1.00 | 19.8 |
  | 2 | 154 | 1.34 | 26.6 |
  | 4 | 141 | 1.47 | 29.1 |

  Every run reproduces the reference descriptors. Each of the 9 jobs pays a fixed start-up cost (JVM,
  Spark session), which limits the speed-up at this data size.

## 2. Check of the model code (FGCS 2026 anchor)

The authors' later offline experiment (800 runs with their own descriptors), run unchanged, gives
exactly the published file to 3 decimals: MAE 163.045, RMSE 517.132, P95 903.808, max 4431.059,
Kendall tau 0.694, precision@10 0.805 (`results/reproduced/fgcs2026_anchor/`). So the model code and my
environment are not where the differences below come from.

## 3. FGCS 2025

AME = mean of the LOOCV and LOGOCV HME (lower is better). DS: Garralda model deterministic, baselines 3
seeds. DS' and DS'': Garralda model 10 seeds, baselines 1 seed. `results/comparison/fgcs2025_vs_ours.csv`
has every metric, with min, max and sd over seeds.

**DS (Table 6)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 136.4 / 107.1 | 293.6 / 223.5 | 273.0 / 215.3 |
| Dou | 229.0 / 206.0 | 234.5 / 227.5 | 319.5 / 299.0 |
| Garralda | 88.5 / 117.0 | 225.4 / 242.5 | 199.9 / 245.3 |

**DS' (Table 8)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 44.1 / 37.2 | 254.8 / 226.9 | 183.4 / 162.5 |
| Dou | 159.9 / 157.6 | 241.4 / 260.5 | 272.8 / 284.1 |
| Garralda | 52.7 / 44.5 | 219.9 / 233.0 | 174.8 / 176.1 |

**DS'' (Table 10)**

| Model | LOOCV MAE paper / ours | LOGOCV MAE paper / ours | AME paper / ours |
|---|---|---|---|
| Prats | 61.1 / 69.6 | 160.6 / 135.0 | 143.5 / 135.4 |
| Dou | 168.8 / 94.2 | 199.8 / 188.6 | 265.7 / 192.8 |
| Garralda (median of 10) | 59.8 / 44.7 | 108.6 / 108.4 | 111.6 / 100.1 |

Garralda on DS'': the mean over the 10 seeds is dominated by the 4 seeds where LOGOCV blows up (mean
LOGOCV MAE 154,358). The other 6 seeds and the median sit right on the paper's numbers.

**Base learner ablation (Table 5, DS, 3 seeds for the random learners)**

| Learner | LOOCV MAE paper / ours | LOGOCV MAE paper / ours |
|---|---|---|
| Random forest | 115.3 / 109.5 | 257.8 / 270.0 |
| AdaBoost | 123.3 / 114.8 | 293.5 / 273.2 |
| Gradient boosting | 115.6 / 106.6 | 269.6 / 264.7 |
| NNLS (the paper's choice) | 88.5 / 117.0 | 225.4 / 242.5 |

NNLS has the lowest LOGOCV error in both, but our LOOCV error is the highest of the four: one svm run
(11,469 s predicted for 4,819 s) dominates it.

**Headline claim.** The paper says its model lowers AME by 18% against Prats and 45% against Dou (31%
overall). Recomputed from its own Table 12 that is 17.9% and 43.8%. Ours, using the median over seeds
(needed for DS''): 34.7% against Dou (17.9% on DS, 38.2% on DS', 48.1% on DS'') but only 1.3% against
Prats (-14.0% on DS, -8.1% on DS', +26.0% on DS''). The model clearly beats Dou, as in the paper, but
not Prats: the Prats baseline does much better in our runs than in the paper (AME 215 against 273 on
DS). Its LOGOCV branch in the published code can't run (I-06), so the paper's Prats numbers came from
code we don't have.

**Sensitivity (DS, AME)**

- lambda (code convention, I-11): 275.7 at 0, 245.3 at 0.5 (default), 193.2 at 1.0. With lambda near 1 our
  AME drops below the paper's 199.9.
- Upper bound of k: 204.6 at 6, 245.3 at 18 (default), 285.6 at 30.
- Descriptor variants: log order 275.0, no Cusum 190.0, Cusum with degree 1 182.1, Cusum with degree 3
  293.0 (default 245.3). The simpler descriptors do better on DS.

**Same code on two CPU types (I-19).** On DS the Garralda model and Dou give the same numbers on the Mac
(ARM64) and the cloud VM (x86-64) at the precision reported here. DS' and DS'' differ in the second or
third significant digit, and Prats on DS has AME 215.2 on ARM64 and 220.0 on x86-64.

## 4. Re-runs on my own cluster

69 runs of 8 HiBench workloads (`docs/cluster.md`), featurised with the same code and compared with the
215 runs of 2022 (`results/reproduced/cluster_rerun/`).

| | Garralda descriptor | Prats descriptor |
|---|---|---|
| Re-runs whose nearest 2022 run is the same workload | 66 / 69 (95.7%) | 63 / 69 (91.3%) |
| ... among the 3 nearest | 97.1% | 92.8% |
| Same check inside the 2022 runs (leave-one-out, same 8 workloads) | 100% | 83.3% |

The Garralda descriptor misses 2 bayes runs (nearest: gbt and pca) and 1 sleep run (nearest: scan).
The Prats descriptor mistakes wordcount for pagerank 6 times.

Distances in the scaled Garralda space (medians): a re-run is 1.38 from its 2022 twin (same workload,
size and setting), about the same as two 2022 runs of one workload under different settings (1.42), and
far from other workloads (3.42). Two repetitions of the same job on the VM are 0.99 apart. The features
that move most between the platforms are timing ones (result serialisation, run time, fetch wait,
deserialisation), which also vary most between repetitions.

Run times on the VM are a median 0.37 of the 2022 times and keep their order (Spearman 0.72 over 45
cells). Feeding the re-runs to the model:

| History used for the prediction | MAE (s) | MAPE | Neighbours from the same workload |
|---|---|---|---|
| 2022 runs | 19.8 | 271% | 63% |
| 2022 runs + VM runs of the other workloads | 14.7 | 194% | 63% |
| VM runs of the other workloads only | 6.6 | 59% | - |

The descriptor transfers across platforms, but the run times don't: the model needs some runs from the
new platform before its predictions are useful there, which is the incremental setting the paper argues
for.
