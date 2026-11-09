# Decisions

Things the papers don't pin down, or where I had to choose between options. Each entry says what I did
and why. The IDs are referred to in the code and in the report.

### D-01 Scope
Reproduce FGCS 2025 starting from the raw event logs: rebuild the descriptor extraction
(not published), check it against the JPDC 2024 results, run the authors' model and the two baselines,
re-run the authors' FGCS 2026 offline experiment as an exact-match check of the model code, and
re-execute a set of HiBench workloads on a cluster I set up myself. The extraction step is the missing
link between the published data and the published numbers, so it had to be rebuilt anyway.

### D-02 Task order inside a series
Tasks are sorted by Task ID (launch order). The paper only says
"ordered sequence of events" and the order in the log (by finish time) is the other obvious reading.
Task ID order gets closer to the per-workload distances in JPDC Table 5 (median absolute difference
0.14, against 0.49 for log order), so I used it. Log order is kept as a variant (`order_log`).

### D-03 Rescaling after the Cusum
The Cusum series is min-max scaled to [0, 1] before the polynomial
fit. Neither paper mentions this step. I worked it out from the authors' own descriptor file in the FGCS
2026 repository (`historical_dataset.json`): the fitted curves stay roughly inside [0, 1], and every
2-point series is stored as exactly (-0.5, -0.5, 1, 0) or (0.5, 0.5, -0.0, 0), which is what this
construction gives and nothing else I tried does. Without the step the dataset diameter is 72,708
instead of the published 27.55.

### D-04 Polynomial fit
x = linspace(0, 1, n), `numpy.polyfit` with degree 2, and the error term is the
MAE of the fit. The x scale comes from the same 2-point evidence as D-03.

### D-05 Constant series
Stored as (0, 0, 0, 0), like in the authors' file.

### D-06 Tasks per stage
Feature 25 is one value per stage, the number of tasks in it, in stage ID order.
The authors' file has (-0.5, -0.5, 1, 0) for applications with 2 stages, so it is a series with one
point per stage.

### D-07 Peak execution memory
Read from the task accumulable `internal.metrics.peakExecutionMemory`.
Spark 2.4.8 doesn't write it into "Task Metrics" (that only started with Spark 3.0).

### D-08 Task events
Every SparkListenerTaskEnd that has task metrics is used (1,251,335 tasks), the 3
failed tasks included. 67 events without metrics are skipped. No guidance in the paper; the effect is
negligible.

### D-09 Building DS''
Each duplicate (the second half of DS') gets its own factor ~U[0.75, 1.25] on its
descriptor, and the duplicates of 12 randomly picked workload types are renamed as new types, which
gives 24 + 12 = 36 groups for LOGOCV. The paper's description ("50% of the workloads in DS' ... each
workload's descriptor vector was multiplied ..., resulting in 36 distinct workload types") can be read
in more than one way; this is the reading that fits all three numbers.

### D-10 Prats descriptor
Mean, min, max and population std of each of the 25 features (100 values),
as in the JPDC "base-approach". The input size used by the Prats model is the total input bytes in GB.

### D-11 Randomness
The authors' random forests have no `random_state` and the paper gives no seeds.
I seed NumPy's global RNG before every unit of work, so every run repeats exactly, and I run several
seeds wherever randomness matters. The DS'/DS'' noise uses `numpy.random.default_rng(seed)`.

### D-12 Configuration vector
The 6 parameters of FGCS Table 4, read from each log: driver cores (never
set, so the Spark default 1), driver memory (1 GB for small/medium, 3 GB for large), executor cores,
executor instances, executor memory, `spark.sql.shuffle.partitions`. Read from the logs, not from the
folder names (see I-02).

### D-13 Target
ApplicationEnd minus ApplicationStart in seconds, not rounded. This matches the
published per-workload statistics within 1.5 s for 15 of the 24 workloads.

### D-14 Neighbourhood bounds
K = [3, 18] on DS, [3, 24] on DS' and [3, 5] on DS'' (paper Tables 5/6,
8, 10). The rule of thumb in Appendix A gives the same 18 and 24. The Dou model uses k = k_max.

### D-15 Baseline grid
The authors' grid runs on scikit-learn 1.4.2, the version their requirements
pin. `max_features='auto'` was removed in scikit-learn 1.3, so only the 'sqrt' half of the grid can be
fitted. I removed 'auto' from the grid (patch for I-09). That gives identical predictions on DS
(checked unit by unit) and stops the Dou model from crashing on DS''.

### D-16 Prats LOGOCV
One random forest per held-out workload group, trained on [descriptor, input
size, configuration], one prediction per held-out run. The published code for this branch can't run
(I-05, I-06); this is what it was clearly meant to do, and it matches the LOOCV branch of the same file.

### D-17 Software versions
NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.2, pandas 2.2.3 (the authors'
requirements say numpy~=1.26, scikit-learn~=1.4, scipy~=1.12), Python 3.10 on the Mac and 3.11 on the
cloud VM, PySpark 3.5 for the featuriser. Spark 2.4.8, Hadoop 2.7.7 and Java 8 for the re-runs, the
same as in the original logs.

### D-18 Failed predictions
Same as the authors' code: if a fold raises (for example the Yeo-Johnson
fit fails) or predicts NaN, that sample is left out of the metric. Every result file records how many.

### D-19 Re-execution cluster
Spark standalone (master and 3 workers) plus HDFS on one Linux VM instead
of the original YARN cluster. Same Spark, Hadoop, Java and HiBench versions and data sizes, same number
of executors, cores and shuffle partitions. Executor memory is divided by 4 because the VM only has
about 4 GB of RAM, the driver gets 512 MB, and `spark.scheduler.minRegisteredResourcesRatio=0.8` (the YARN
default) so jobs wait for their executors like on YARN. Details and the exceptions are in
`docs/cluster.md`.

### D-20 Campaign 2 sizing (VM 2)
VM 2 has 8 vCPUs and 10 GB of RAM, so campaign 2 runs with executor memory divided by 2 instead of 4
(MEM_DIV=2: 2 GB for setting 1, 1.5 GB for settings 2 and 3), worker daemons at 2 GB, driver at 1 GB,
and the same 3 workers x 4 cores as campaign 1. Campaign 2 writes to `data/cluster2/` with benchmark
group id `rerun-vm2` (campaign 1: `rerun-vm`), so runs of the two campaigns can't mix. Same plan
generator and seed as campaign 1 (8 workloads, settings 1-3, tiny x2 and small x1), with bayes small
back in because the memory that killed it on VM 1 is no longer the limit.
