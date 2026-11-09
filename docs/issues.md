# Problems found

Problems in the published material (data, code, papers) and in reproducing it, with the evidence and
what I did about each one. Severity: *blocking* means it stops the published material from running
or being reproduced as it is, *material* means it changes the numbers, *minor* means it doesn't.

### I-01 One run missing (data, minor)
215 event logs are published but the paper uses 216: gbt / large / setting 4 is missing
(`results/reproduced/data_audit/summary.json`). Everything runs on the 215.

### I-02 Folder names don't match the runs (data, minor)
The folder `big_executor3_cores3_memory3g_shuffle10` ran with `spark.sql.shuffle.partitions=100`, not
10, and the inner folder prefixes (small/large/big) don't line up with the paper's small/medium/large.
The configuration is read from the logs instead (D-12).

### I-03 Missing import in `algorithm.py` (code, blocking)
`Tuple` is used without being imported, so importing the module fails. Patched
(`patches/authors_code.diff`).

### I-04 Wrong import path in `validation.py` (code, blocking)
It imports `performance.domain.model.garralda_perf_model`, which isn't in the repository. Patched to
import from `algorithm`.

### I-05 IndentationError in `baselines/prats_perf_model.py` (code, blocking)
Line 171. Patched.

### I-06 Prats LOGOCV branch can't run (code, blocking)
It calls the model builder with keyword arguments it doesn't accept and predicts the whole held-out
group as one 3-D input, so it can't have produced the published Prats LOGOCV numbers. The patch does
what the branch was meant to do (D-16).

### I-07 The published logs are not exactly the runs behind the tables (data vs paper, material)
Per-workload run-time statistics match FGCS Table 3 within 1.5 s for 15 of 24 workloads, but 8 differ a
lot (for example sort max 281 s against 35 s published, join 200 s against 88 s, aggregation max 75 s
against 124 s). JPDC says each workload was run several times and one run kept, so the published files
are probably other repetitions. This is the most likely reason the DS numbers don't match
(`results/reproduced/data_audit/t_stats_vs_paper.csv`).

### I-08 Aggregate std in FGCS Table 3 (paper, minor)
The table gives 576 s (the text says 575) but its own per-workload rows give a pooled std of about
770 s (ours is 783 s).

### I-09 `max_features='auto'` in the baseline grids (code, material for the baselines)
scikit-learn removed 'auto' in 1.3, and the authors pin 1.4. On DS and DS' half of the grid just fails
with a warning and is skipped, so the baselines depend on the library version. On DS'' it is worse:
the Dou model uses only k = 5 neighbours, the 5-fold grid search then gets a NaN score for every
candidate, the refit falls back to the first grid point, which is an 'auto' one, and every prediction
fails. I removed 'auto' from both grids (patch). DS predictions are unchanged (checked unit by unit).

### I-10 Random forests without a seed (code, material for the baselines)
The baselines and the RF/AdaBoost/GB learners of Table 5 have no `random_state`, so results change from
run to run. Seeded runs and several seeds (D-11).

### I-11 lambda applied the other way round (code vs paper, minor)
Paper Eq. 7 is lambda*CdV + (1-lambda)*CqC, the code computes lambda*CqC + (1-lambda)*CdV. Same result
at the default 0.5, but the lambda axis of Appendix B is mirrored. My lambda sweep uses the code's
convention.

### I-12 `k_bound_heuristics` uses min instead of max (code vs paper, minor)
The code has `min(3, ...)` where the paper (Eq. A.1a) has `max(3, ...)`; with n = 216 the code would give
k_min = 1. The published experiments pass the bounds explicitly, so it has no effect there.

### I-13 NNLS isn't quite NNLS (code vs paper, minor)
The paper's Eq. 10 has no intercept. The code uses `LinearRegression(positive=True)` with a free
intercept, a Yeo-Johnson transform of the target and min-max scaled features. Only the coefficients
are kept non-negative, so negative predictions are possible (0 to 2 per run in our results).

### I-14 Yeo-Johnson fit sometimes fails (code, minor)
On some neighbour sets SciPy raises a BracketError and the authors' loops drop the sample. 0 to 1 of the
430 LOOCV folds per DS' seed. Kept and counted (D-18).

### I-15 Typos in the published tables (paper, minor)
Some HME and AME values don't follow from the printed MAE and RMSE (Eq. 13, 14), e.g. Dou's LOGOCV HME
on DS'' is printed as "248.47.88". See `results/comparison/paper_table_audit.csv`.

### I-16 Descriptor extraction not published (featuriser, blocking)
Neither repository has the code that turns event logs into descriptors (`spark-self-tuning-framework`
imports a `parser` package that isn't there). Rebuilt from the JPDC 2024 description and checked against
its results (D-02 to D-08).

### I-17 JPDC Table 5 ratio (paper, minor)
LDA medium is printed as 0.16 for a distance of 3.69, but 3.69 / 27.55 = 0.13.

### I-18 DS'' LOGOCV is numerically unstable (code, material)
In 4 of 10 DS'' seeds a few LOGOCV predictions explode (largest errors 3e5 to 3e8 s) and take the HME
up to 1e3 to 1e6 s. With only 3 to 5 neighbours the linear model has to extrapolate to new
configurations, and the inverse Yeo-Johnson transform blows the extrapolated value up. The other 6
seeds are close to the paper. I report the median over seeds as well as the mean.

### I-19 Small differences between x86-64 and ARM64 (environment, minor)
Same code, versions and seeds give identical results when repeated on the same machine, but the cloud
VM (x86-64) and the Mac VM (ARM64) differ slightly. The Garralda descriptors differ in the last digits
(the least squares fit goes through LAPACK, which is built differently for each CPU), and the random
forest baselines can pick different trees in near ties: Prats on DS has AME 219.96 on x86-64 and
215.20 on ARM64. The Prats descriptors and the run metadata are bit-for-bit identical on both.
