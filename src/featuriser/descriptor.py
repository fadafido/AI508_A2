"""Descriptor maths, shared by the plain Python and the PySpark featurisers.

Garralda descriptor (JPDC 189 (2024) 104881, Sec. 4.1), for each task feature series (one value per task):
    tf' = min-max(Tf)                 JPDC Eq. 5
    z   = (tf' - mean) / std          Eq. 6
    z'  = cumulative sum of z         Eq. 7 (Cusum)
    y   = min-max(z') to [0, 1]       not in the paper, see D-03 in docs/decisions.md
    x   = linspace(0, 1, n)           D-04
    fit y ~ a x^2 + b x + c and keep a, b, c and the MAE of the fit (Eq. 8, 9a)
25 features x 4 values = 100 numbers. A constant series gives zeros (D-05).

Prats descriptor (JPDC Sec. 5.3, the "base-approach"): mean, min, max and std of each feature, 25 x 4.

The features are in the same order as in the authors' descriptor file (historical_dataset.json in
spark-self-tuning-framework), so their 17-feature YORO subset can be picked by index.
"""
from __future__ import annotations

import warnings

import numpy as np

# (name, path inside the SparkListenerTaskEnd event); "acc:" = Task Info accumulable (Spark 2.4 logs keep
# peak execution memory only there); "derived:" = computed from the stage structure.
TASK_FEATURES = [
    ("Result Size", ("Task Metrics", "Result Size")),
    ("Memory Bytes Spilled", ("Task Metrics", "Memory Bytes Spilled")),
    ("Disk Bytes Spilled", ("Task Metrics", "Disk Bytes Spilled")),
    ("Shuffle Read: Remote Blocks Fetched", ("Task Metrics", "Shuffle Read Metrics", "Remote Blocks Fetched")),
    ("Shuffle Read: Local Blocks Fetched", ("Task Metrics", "Shuffle Read Metrics", "Local Blocks Fetched")),
    ("Shuffle Read: Remote Bytes Read", ("Task Metrics", "Shuffle Read Metrics", "Remote Bytes Read")),
    ("Shuffle Read: Remote Bytes Read To Disk", ("Task Metrics", "Shuffle Read Metrics", "Remote Bytes Read To Disk")),
    ("Shuffle Read: Local Bytes Read", ("Task Metrics", "Shuffle Read Metrics", "Local Bytes Read")),
    ("Shuffle Read: Records Read", ("Task Metrics", "Shuffle Read Metrics", "Total Records Read")),
    ("Shuffle Write: Bytes Written", ("Task Metrics", "Shuffle Write Metrics", "Shuffle Bytes Written")),
    ("Shuffle Write: Records Written", ("Task Metrics", "Shuffle Write Metrics", "Shuffle Records Written")),
    ("Input: Bytes Read", ("Task Metrics", "Input Metrics", "Bytes Read")),
    ("Input: Records Read", ("Task Metrics", "Input Metrics", "Records Read")),
    ("Output: Bytes Written", ("Task Metrics", "Output Metrics", "Bytes Written")),
    ("Output: Records Written", ("Task Metrics", "Output Metrics", "Records Written")),
    ("Executor Deserialize Time", ("Task Metrics", "Executor Deserialize Time")),
    ("Executor Deserialize CPU Time", ("Task Metrics", "Executor Deserialize CPU Time")),
    ("Executor Run Time", ("Task Metrics", "Executor Run Time")),
    ("Executor CPU Time", ("Task Metrics", "Executor CPU Time")),
    ("JVM GC Time", ("Task Metrics", "JVM GC Time")),
    ("Result Serialization Time", ("Task Metrics", "Result Serialization Time")),
    ("Shuffle Write: Write Time", ("Task Metrics", "Shuffle Write Metrics", "Shuffle Write Time")),
    ("Shuffle Read: Fetch Wait Time", ("Task Metrics", "Shuffle Read Metrics", "Fetch Wait Time")),
    ("Peak Execution Memory", ("acc", "internal.metrics.peakExecutionMemory")),
    ("Tasks per Stage", ("derived", "tasks_per_stage")),
]
FEATURE_NAMES = [n for n, _ in TASK_FEATURES]
N_TASK_METRICS = 24          # the first 24 come from each task event; the 25th is per stage
YORO_17_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 15, 16, 17, 18, 19, 20, 22]   # authors' pick_17_features_from_garralda_vector


def task_metric_values(event: dict) -> list[float]:
    """The 24 per-task values of one SparkListenerTaskEnd event (missing -> 0)."""
    tm = event.get("Task Metrics") or {}
    acc = {}
    for a in (event.get("Task Info") or {}).get("Accumulables", []) or []:
        if "Name" in a and "Update" in a:
            acc[a["Name"]] = a["Update"]
    out = []
    for _, path in TASK_FEATURES[:N_TASK_METRICS]:
        if path[0] == "acc":
            v = acc.get(path[1], 0)
        else:
            v = tm
            for k in path[1:]:
                v = v.get(k, 0) if isinstance(v, dict) else 0
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo)


def garralda_feature(series, degree: int = 2, cusum: bool = True, final_minmax: bool = True) -> list[float]:
    """Polynomial coefficients (highest power first) + MAE of the fit, for one task-feature series."""
    y = np.asarray(series, dtype=float)
    if y.size == 0 or np.ptp(y) == 0:
        return [0.0] * (degree + 2)
    t = minmax(y)                              # Eq. 5
    sd = t.std()
    z = (t - t.mean()) / sd                    # Eq. 6 (population std; scale cancels after the final min-max)
    if cusum:
        z = np.cumsum(z)                       # Eq. 7
    if final_minmax:
        if np.ptp(z) == 0:
            return [0.0] * (degree + 2)
        z = minmax(z)
    x = np.linspace(0.0, 1.0, z.size)
    with warnings.catch_warnings():            # n <= degree (e.g. 2 stages) is rank-deficient by design
        warnings.simplefilter("ignore", getattr(np, "RankWarning", None) or np.exceptions.RankWarning)
        coef = np.polyfit(x, z, degree)        # least squares; min-norm when n <= degree (as numpy does)
    mae = float(np.mean(np.abs(np.polyval(coef, x) - z)))   # Eq. 8
    return [float(c) for c in coef] + [mae]


def garralda_descriptor(task_matrix: np.ndarray, tasks_per_stage: np.ndarray, degree: int = 2,
                        cusum: bool = True, final_minmax: bool = True) -> list[float]:
    """task_matrix: (n_tasks, 24) in the chosen task order; tasks_per_stage: counts ordered by stage id."""
    vec: list[float] = []
    for j in range(N_TASK_METRICS):
        vec += garralda_feature(task_matrix[:, j] if task_matrix.size else [], degree, cusum, final_minmax)
    vec += garralda_feature(tasks_per_stage, degree, cusum, final_minmax)
    return vec


def prats_descriptor(task_matrix: np.ndarray, tasks_per_stage: np.ndarray) -> list[float]:
    """mean, min, max, std (population) of every task-feature (25 x 4)."""
    vec: list[float] = []
    cols = [task_matrix[:, j] for j in range(N_TASK_METRICS)] + [np.asarray(tasks_per_stage, float)]
    for v in cols:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            vec += [0.0, 0.0, 0.0, 0.0]
        else:
            vec += [float(v.mean()), float(v.min()), float(v.max()), float(v.std())]
    return vec


def descriptor_columns(prefix: str, degree: int = 2) -> list[str]:
    parts = [f"p{k}" for k in range(degree, -1, -1)] + ["mae"] if prefix == "g" else ["mean", "min", "max", "std"]
    return [f"{prefix}{i:02d}_{p}" for i in range(len(FEATURE_NAMES)) for p in parts]
