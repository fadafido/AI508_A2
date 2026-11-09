# Re-running HiBench workloads on my own cluster

The extra experiment: run some of the paper's workloads again on a cluster I set up, extract their
descriptors with my featuriser and check whether they still look like the original 2022 runs.

## Set-up

Everything ran inside the Linux VM on my MacBook Pro (Ubuntu 22.04, ARM64, 4 vCPUs, 3.8 GB RAM, no swap).

| | Original runs (from their event logs) | My re-runs |
|---|---|---|
| Spark | 2.4.8 on YARN, client mode | 2.4.8, standalone master + 3 worker daemons |
| Storage | HDFS (Hadoop 2.7.7) | HDFS 2.7.7, one NameNode + one DataNode, replication 1 |
| Java | 1.8.0_342 | Temurin 1.8.0_512 (aarch64) |
| HiBench | 8.0-SNAPSHOT | same, built from master @ bf390d2 (last commit before the runs) |
| Data generation | HiBench prepare scripts | the same generators, MapReduce in local mode |
| Executors / cores / shuffle partitions | settings 1-3 of JPDC Table 3 | the same |
| Executor memory | 4 GB (setting 1), 3 GB (2 and 3) | divided by 4: 1 GB and 768 MB |
| Driver memory | 1 GB | 512 MB |

The scripts are in `scripts/cluster/` (`setup_tools.sh`, `start.sh`, `hibench.sh`, `make_plan.py`,
`step.sh`). `hibench.sh` does what HiBench's own prepare/run scripts do, with the same data sizes and
parameters, but without HiBench's python2/YARN configuration layer.

Workloads: sort, wordcount, terasort, repartition, sleep, pagerank, bayes and kmeans, at HiBench sizes
"tiny" and "small", which is what the paper calls small and medium (the original logs record the HiBench
size in `spark.app.benchmark.data.size`). Each workload ran under settings 1, 2 and 3; the tiny size twice
(so I can see run-to-run noise), the small size once.

## Things I had to work around

- The VM's hostname doesn't resolve and the NameNode refuses to start without it. `step.sh` runs the
  whole session in a private UTS namespace (`unshare -r -u`) where the hostname is `localhost`.
- The VM stops every process when a shell session ends and a session can't run longer than 3 minutes.
  So the campaign runs in short resumable sessions: start HDFS and Spark (about 20 s), run the pending
  jobs from `data/cluster/plan.txt` while time is left, stop everything. Progress is in
  `data/cluster/progress.txt`, one line per job.
- Spark standalone copies the 118 MB HiBench jar into the work directory of every executor. That filled
  the VM disk after about 15 applications, so the work directory of each run is deleted when it ends.
- With the standalone defaults a job can start before its executors have registered. ScalaSleep then
  ran 2 tasks instead of one per core (4 or 9 in the 2022 logs). Setting
  `spark.scheduler.minRegisteredResourcesRatio=0.8` (the YARN default) fixed it. The first batch of tiny
  runs made before that change is kept in `data/cluster/superseded/` and not used.
- Memory. Three small-size jobs had executors killed by the out-of-memory killer (kmeans under settings
  2 and 3, repartition under setting 2). They were re-run with one worker daemon instead of three and
  executor memory divided by 6 (512 MB). bayes at the small size failed every time (the driver ran out
  of heap at 512 MB, and executors at 512, 682 and 768 MB); it would need more RAM than the VM has, so
  it is not in the results.

## What came out

69 successful runs: 48 at the tiny size (8 workloads x 3 settings x 2) and 21 at the small size (7
workloads x 3 settings). Event logs are in `data/cluster/eventlogs/<size>/setting<N>/`, one record per run
in `data/cluster/runs.jsonl` (including the failed attempts), driver output in `data/cluster/driver_logs/`.

Compared with the 2022 logs:

- Task counts are the same for sort, terasort, repartition, wordcount, kmeans and sleep. pagerank has
  half the tasks (8 instead of 16, 12 instead of 24) because my generated input has 2 partitions where
  the original had 4. bayes at the tiny size has 56 tasks under setting 3 instead of 51.
- randomtextwriter (sort and wordcount inputs) wrote about twice as many bytes as in the original runs
  (wordcount medium: 164 MB read against 82 MB). The original cluster probably compressed the generator
  output or ran more maps; I can't tell from the logs. terasort, repartition, kmeans and pagerank read
  about the same amount of data as before, bayes 10 to 45% more.
- Runs are faster on the VM (median 0.37 of the 2022 time): no YARN container start-up, faster cores and
  everything on one machine. The ranking is kept fairly well (Spearman 0.72 over 45
  workload/size/setting cells).

The descriptor comparison itself is in `src/analysis/cluster_rerun.py` and `docs/results.md`.
