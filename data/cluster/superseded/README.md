First tiny runs (sort, wordcount, terasort, repartition, sleep, pagerank), 9 Nov 2026 12:10-12:22 UTC.
They ran with the Spark standalone default spark.scheduler.minRegisteredResourcesRatio=0, so jobs could
start before the executors had registered (sleep got 2 tasks instead of one per core). Re-run with 0.8
(the YARN default) afterwards. Kept for reference only, not used in the analysis.
