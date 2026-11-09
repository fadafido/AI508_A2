"""Checks the published FGCS 2025 tables against themselves and against the authors' results files.

1. HME in Tables 6, 8 and 10 against HME recomputed from the printed MAE and RMSE (Eq. 13)
2. AME in Table 12 against the printed HMEs (Eq. 14)
3. AMR in Tables 7, 9 and 11 against the mean of the LOOCV and LOGOCV columns
4. the tables against the authors' own results files (results/*-perf-model in their repository)
5. the improvement percentages quoted in Sec. 7.2 and 7.4
Output: results/comparison/paper_table_audit.csv and .json
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.paths import PAPER_REF, COMPARISON, GARRALDA_REPO  # noqa: E402

TOL = 0.015   # printed values have 2 decimals


def hme(mae, rmse):
    return 2 * mae * rmse / (mae + rmse)


def parse_repo_results(path: Path) -> dict:
    """{dataset_index: {"LOOCV": {...}, "LOGOCV": {...}, "AME": x}} from the authors' text results file."""
    out, ds, cv = {}, None, None
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\d)º Experimental", line)
        if m:
            ds = int(m.group(1))
            out[ds] = {}
            continue
        if ds is None:
            continue
        m = re.match(r"\s*(LOOCV|LOGOCV): Evaluation Metrics", line)
        if m:
            cv = m.group(1)
            out[ds][cv] = {}
            continue
        m = re.match(r"\s*(MAE|RMSE|HME|Residual Max|Residual Median|Residual Std\. Dev\.):\s*([\d.]+)", line)
        if m and cv:
            out[ds][cv][m.group(1)] = float(m.group(2))
        m = re.match(r"\s*AME:\s*([\d.]+)", line)
        if m:
            out[ds]["AME"] = float(m.group(1))
            cv = None
        if line.strip().startswith("Results:") and out.get(ds, {}).get("AME"):
            ds = None
    return out


def main():
    P = json.loads((PAPER_REF / "fgcs2025_published.json").read_text())
    rows = []
    for tname, dsn in [("table6_DS", "DS"), ("table8_DSprime", "DS'"), ("table10_DSdoubleprime", "DS''")]:
        for model, v in P[tname].items():
            if model.startswith("_"):
                continue
            for k, cv in [(0, "LOOCV")]:      # LOGOCV HME is the mean of per-group HMEs: not derivable from MAE/RMSE
                rec = hme(v[k], v[k + 1])
                rows.append({"check": "HME from MAE,RMSE", "table": tname, "dataset": dsn, "model": model, "cv": cv,
                             "printed": v[k + 2], "recomputed": round(rec, 2), "consistent": abs(rec - v[k + 2]) <= TOL})
    for i, (dsn, tname) in enumerate([("DS", "table6_DS"), ("DS_prime", "table8_DSprime"), ("DS_doubleprime", "table10_DSdoubleprime")]):
        for j, model in enumerate(["Prats", "Dou", "Garralda"]):
            v = P[tname][model]
            rec = (v[2] + v[5]) / 2
            printed = P["table12_AME"][dsn][j]
            rows.append({"check": "AME from HMEs", "table": "table12_AME", "dataset": dsn, "model": model, "cv": "-",
                         "printed": printed, "recomputed": round(rec, 2), "consistent": abs(rec - printed) <= TOL})
    for tname in ["table7_residuals_DS", "table9_residuals_DSprime", "table11_residuals_DSdoubleprime"]:
        for model, v in P[tname].items():
            if model.startswith("_"):
                continue
            for s, stat in enumerate(["max", "median", "std"]):
                rec = (v[s] + v[3 + s]) / 2
                rows.append({"check": "AMR = mean(LOOCV, LOGOCV)", "table": tname, "dataset": tname.split("_")[-1],
                             "model": model, "cv": stat, "printed": v[6 + s], "recomputed": rec,
                             "consistent": abs(rec - v[6 + s]) <= 1.0})
    # paper vs repository results files
    repo = {m: parse_repo_results(GARRALDA_REPO / "results" / f"{f}-perf-model")
            for m, f in [("Garralda", "garralda"), ("Dou", "dou"), ("Prats", "prats")]}
    for i, (dsn, tname, rtab) in enumerate([("DS", "table6_DS", "table7_residuals_DS"),
                                            ("DS'", "table8_DSprime", "table9_residuals_DSprime"),
                                            ("DS''", "table10_DSdoubleprime", "table11_residuals_DSdoubleprime")], start=1):
        for model in ["Prats", "Dou", "Garralda"]:
            r = repo[model].get(i, {})
            v = P[tname][model]
            rr = P[rtab][model]
            for k, cv in [(0, "LOOCV"), (3, "LOGOCV")]:
                for off, met in enumerate(["MAE", "RMSE", "HME"]):
                    if met in r.get(cv, {}):
                        rows.append({"check": "paper vs repo results file", "table": tname, "dataset": dsn, "model": model,
                                     "cv": f"{cv} {met}", "printed": v[k + off], "recomputed": r[cv][met],
                                     "consistent": abs(r[cv][met] - v[k + off]) <= TOL})
                for off, met in enumerate(["Residual Max", "Residual Median", "Residual Std. Dev."]):
                    if met in r.get(cv, {}):
                        pv = rr[(0 if cv == "LOOCV" else 3) + off]
                        rows.append({"check": "paper vs repo results file", "table": rtab, "dataset": dsn, "model": model,
                                     "cv": f"{cv} {met}", "printed": pv, "recomputed": r[cv][met],
                                     "consistent": abs(r[cv][met] - pv) <= 1.0})
            if "AME" in r:
                pv = P["table12_AME"][["DS", "DS_prime", "DS_doubleprime"][i - 1]][["Prats", "Dou", "Garralda"].index(model)]
                rows.append({"check": "paper vs repo results file", "table": "table12_AME", "dataset": dsn, "model": model,
                             "cv": "AME", "printed": pv, "recomputed": r["AME"], "consistent": abs(r["AME"] - pv) <= TOL})
    df = pd.DataFrame(rows)
    COMPARISON.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMPARISON / "paper_table_audit.csv", index=False)
    # headline claims (Sec. 7.4): improvements of overall AME
    ame = P["table12_AME"]
    tot = {m: sum(ame[d][j] for d in ["DS", "DS_prime", "DS_doubleprime"]) for j, m in enumerate(["Prats", "Dou", "Garralda"])}
    per_ds = {m: [round(100 * (1 - ame[d][2] / ame[d][j]), 1) for d in ["DS", "DS_prime", "DS_doubleprime"]]
              for j, m in [(0, "Prats"), (1, "Dou")]}
    claims = {"AME_sum": tot,
              "per_dataset_improvement_pct": per_ds,
              "mean_of_per_dataset_improvement_pct": {m: round(sum(v) / 3, 1) for m, v in per_ds.items()},
              "improvement_vs_Prats_pct": round(100 * (1 - tot["Garralda"] / tot["Prats"]), 1),
              "improvement_vs_Dou_pct": round(100 * (1 - tot["Garralda"] / tot["Dou"]), 1),
              "paper_says": P["headline_claims"]}
    summ = {"checks": int(len(df)), "inconsistent": int((~df.consistent).sum()),
            "inconsistent_rows": df[~df.consistent].to_dict("records"), "headline_claims": claims}
    (COMPARISON / "paper_table_audit.json").write_text(json.dumps(summ, indent=2))
    print(df[~df.consistent].to_string(index=False))
    print(json.dumps(claims, indent=2))


if __name__ == "__main__":
    main()
