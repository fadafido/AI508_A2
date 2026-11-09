"""The datasets of FGCS 2025 Sec. 6.1 and 7.3, built from our descriptors.

DS    215 runs (the paper has 216). W1 = Garralda descriptor, W2 = Prats descriptor, CS = the 6
      configuration parameters of Table 4, T = run time.
DS'   DS plus a copy of every run with Gaussian noise on the time (sd = t*p, p ~ U[0.05, 0.10]);
      3% of the copies get 3 times the sd, as outliers (Sec. 7.3.1, Eqs. 15-16).
DS''  DS' where each copy's descriptor is multiplied by its own factor ~U[0.75, 1.25] and the copies
      of 12 random workload types become new types, 36 types in total (Sec. 7.3.2; the description is
      ambiguous, see D-09 in docs/decisions.md).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PROCESSED  # noqa: E402

CS_COLS = ["driver_cores", "driver_memory_gb", "executor_cores", "executor_instances", "executor_memory_gb",
           "sql_shuffle_partitions"]
K_RANGES = {"DS": (3, 18), "DSp": (3, 24), "DSpp": (3, 5)}       # FGCS Tables 5/6, 8, 10


def load_ds(variant: str = "default") -> dict:
    apps = pd.read_csv(PROCESSED / "apps.csv")
    G = pd.read_csv(PROCESSED / f"garralda__{variant}.csv").set_index("app").loc[apps.app].values
    P = pd.read_csv(PROCESSED / "prats.csv").set_index("app").loc[apps.app].values
    return {
        "name": "DS", "app": apps.app.values, "group": apps.workload.values.astype(object),
        "W1": G.astype(float), "W2": P.astype(float), "CS": apps[CS_COLS].values.astype(float),
        "T": apps.exec_time_s.values.astype(float),
        "ids": (apps.input_bytes.values.astype(float) / 1024 ** 3).reshape(-1, 1),   # input data size (GB)
        "duplicate": np.zeros(len(apps), bool), "k_range": K_RANGES["DS"],
    }


def make_ds_prime(ds: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(ds["T"])
    t = ds["T"]
    p = rng.uniform(0.05, 0.10, n)
    sd = t * p
    out_idx = rng.choice(n, size=int(round(0.03 * n)), replace=False)
    sd[out_idx] *= 3.0                                               # gamma = 3
    t_dup = t + rng.normal(0.0, sd)
    d = {k: (np.concatenate([v, v]) if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
    d["T"] = np.concatenate([t, t_dup])
    d["app"] = np.concatenate([ds["app"], np.array([a + "#dup" for a in ds["app"]], dtype=object)])
    d["duplicate"] = np.concatenate([np.zeros(n, bool), np.ones(n, bool)])
    d["outlier"] = np.zeros(2 * n, bool)
    d["outlier"][n + out_idx] = True
    d["name"], d["k_range"], d["seed"] = "DSp", K_RANGES["DSp"], seed
    return d


def make_ds_doubleprime(dsp: dict, seed: int) -> dict:
    rng = np.random.default_rng(10_000 + seed)
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in dsp.items()}
    dup = np.where(d["duplicate"])[0]
    f = rng.uniform(0.75, 1.25, len(dup))
    d["W1"][dup] *= f[:, None]
    d["W2"][dup] *= f[:, None]
    types = np.array(sorted(set(dsp["group"])))
    new_types = set(rng.choice(types, size=len(types) // 2, replace=False))
    for i in dup:
        if d["group"][i] in new_types:
            d["group"][i] = d["group"][i] + "_new"
    d["name"], d["k_range"], d["seed"] = "DSpp", K_RANGES["DSpp"], seed
    d["new_types"] = sorted(new_types)
    return d


def build(name: str, seed: int = 0, variant: str = "default") -> dict:
    ds = load_ds(variant)
    if name == "DS":
        return ds
    dsp = make_ds_prime(ds, seed)
    return dsp if name == "DSp" else make_ds_doubleprime(dsp, seed)
