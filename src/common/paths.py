"""Paths used across the project, all relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "external"
DATA = ROOT / "data"
RAW = DATA / "raw"
LOGS_DIR = RAW / "eventlogs"            # extracted Spark event logs (2022 HiBench runs, 4.09 GB)
INTERIM = DATA / "interim"
SERIES = INTERIM / "series"             # per-application task series (npz) from the reference parser
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
REPRODUCED = RESULTS / "reproduced"
PAPER_REF = RESULTS / "paper_reference"
COMPARISON = RESULTS / "comparison"
FIGURES = RESULTS / "figures"
RUN_LOGS = ROOT / "logs"

GARRALDA_REPO = EXTERNAL / "garralda-performance-model"
STF_REPO = EXTERNAL / "spark-self-tuning-framework"

for _p in (RAW, INTERIM, SERIES, PROCESSED, REPRODUCED, COMPARISON, FIGURES, RUN_LOGS):
    _p.mkdir(parents=True, exist_ok=True)
