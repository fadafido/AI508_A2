# Environment

## Machines

The final results come from the MacBook. A cloud VM was used first to develop the pipeline and later for
the slow baseline runs that need many CPU hours (DS' and DS'', and the extra seeds on DS). Every result
file stores the machine it ran on (`env` field), and the comparison table has a `machines` column.

| | Mac (main) | Cloud VM |
|---|---|---|
| Hardware | MacBook Pro, Apple M3 Pro | Intel Xeon @ 2.10 GHz |
| Where the code ran | Linux VM on the Mac: Ubuntu 22.04.5, kernel 6.8, ARM64, 4 vCPUs, 3.8 GB RAM | Ubuntu 24.04.5, x86-64, 2 vCPUs, 7.8 GB RAM |
| Python | 3.10.12 | 3.11.18 |
| Java (for PySpark) | OpenJDK 11.0.32 | OpenJDK 21.0.12 |
| Java (for the cluster) | Temurin 1.8.0_512 | Temurin 1.8.0_512 (test runs only) |

Full captures: `results/env/mac_m3pro_linux_vm.txt` and `results/env/cloud_x86_vm.txt`
(`env/capture_env.sh`).

## Python packages

Main environment (`env/requirements.txt`, the versions in the authors' requirements file plus PySpark):
NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.2, pandas 2.2.3, joblib 1.4.2, matplotlib 3.8.4,
PySpark 3.5.9. Same versions on both machines.

The FGCS 2026 anchor uses its own environment with the versions that repository declares
(`env/requirements-fgcs2026.txt`: SciPy 1.15.3, scikit-learn 1.6.1, CatBoost, pydantic, mongomock and a
few more). `env/setup_env.sh` creates both virtual environments.

## External code and data

| What | Source | Version |
|---|---|---|
| Model, baselines, metrics, raw event logs | github.com/mgarralda/garralda-performance-model (MIT) | commit 9f23b627 (2026-05-18) |
| FGCS 2026 framework and the authors' descriptor file | github.com/mgarralda/spark-self-tuning-framework (CC BY-NC 4.0) | commit 811b780a (2026-05-18) |
| HiBench | github.com/Intel-bigdata/HiBench (Apache 2.0) | commit bf390d2 (2022-01-10), 8.0-SNAPSHOT |
| Spark | archive.apache.org, spark-2.4.8-bin-hadoop2.7.tgz | sha256 feedb945...96ce91cd |
| Hadoop | archive.apache.org, hadoop-2.7.7.tar.gz | sha256 d129d08a...9f9c2026 |
| JDK 8 | Adoptium Temurin 8u512 aarch64 | sha256 01263df2...f7c69aaa |

`scripts/fetch_external.sh` clones the two repositories at these commits into `external/`. The patched
copy of the authors' code is generated in `build/` by `patches/apply_patches.py` (diff in
`patches/authors_code.diff`). `external/SOURCES.md` has the full commit hashes, `tools/downloads.sha256`
the full checksums of the cluster software, and `tools/hibench/SHA256SUMS` those of the two HiBench
jars.

The raw data is the split zip in the authors' repository (`resources/all zip.zip.001` to `.007`).
`src/data/prepare_raw.py` joins and extracts it: 4,092,562,038 bytes, 215 event logs, zip sha256
967c91bf...e7606aeb, CRC check passes (`data/raw/MANIFEST.json`). The raw logs are not in git.

## Reproducing everything

```
bash env/setup_env.sh                 # two virtual environments
bash scripts/run_all.sh               # data, featuriser, validation, experiments, comparison
bash scripts/cluster/setup_tools.sh   # only for the re-runs: JDK 8, Spark 2.4.8, Hadoop 2.7.7
```

Most of the run time is the two baselines on DS' and DS'': on the 2-core cloud VM, Dou took 27 minutes
and Prats 45 minutes on DS'. Everything else takes a few minutes.
