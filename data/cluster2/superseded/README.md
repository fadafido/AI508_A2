Pagerank runs of the first campaign-2 session, 9 Nov 2026 17:19-17:32 UTC.
Their data generation had failed (the HiBench generator creates DistributedCache symlinks in the
current directory, which was the shared project mount, and that mount refuses symlinks), so these
runs read a 4-byte input and finished in about 3 s instead of the expected time. hibench.sh now
changes into the work folder on the VM's own disk before running anything, and the pagerank jobs
were re-run with properly generated input. The failed bayes and pagerank prepare/run records from
the same session are in progress.txt here too (their failure counters were reset for the re-run).
Kept for reference only, not used in the analysis.
