"""PySpark featuriser: raw event logs to Garralda and Prats descriptors, configuration and run time.

All the heavy work stays in Spark SQL, nothing is collected per application on the driver:
  1. read the logs as text, keep the SparkListenerTaskEnd lines and parse the 24 task metrics with
     from_json (peak execution memory comes from the accumulables, where Spark 2.4 puts it)
  2. unpivot to long format (application, metric, task position, value), about 30 million rows
  3. window functions per (application, metric): standardise, running sum (Cusum), min-max, and
     x = i / (n - 1), the same as numpy.linspace(0, 1, n)
  4. quadratic least squares per group from summed moments. The 3x3 normal equations are solved for
     all groups at once (groups with n <= 3 go through numpy.polyfit like the plain version), then a
     second pass gives the MAE of each fit
  5. Prats descriptor from mean/min/max/std aggregates, metadata from the other events

    python src/featuriser/spark_featuriser.py --master local[4] [--out DIR] [--glob PATTERN]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.featuriser.descriptor import TASK_FEATURES, N_TASK_METRICS  # noqa: E402

from pyspark.sql import SparkSession, Window, functions as F, types as T  # noqa: E402

L = T.LongType()
TM_SCHEMA = T.StructType([
    T.StructField("Executor Deserialize Time", L), T.StructField("Executor Deserialize CPU Time", L),
    T.StructField("Executor Run Time", L), T.StructField("Executor CPU Time", L), T.StructField("Result Size", L),
    T.StructField("JVM GC Time", L), T.StructField("Result Serialization Time", L),
    T.StructField("Memory Bytes Spilled", L), T.StructField("Disk Bytes Spilled", L),
    T.StructField("Peak Execution Memory", L),
    T.StructField("Shuffle Read Metrics", T.StructType([T.StructField(n, L) for n in [
        "Remote Blocks Fetched", "Local Blocks Fetched", "Fetch Wait Time", "Remote Bytes Read",
        "Remote Bytes Read To Disk", "Local Bytes Read", "Total Records Read"]])),
    T.StructField("Shuffle Write Metrics", T.StructType([T.StructField(n, L) for n in [
        "Shuffle Bytes Written", "Shuffle Write Time", "Shuffle Records Written"]])),
    T.StructField("Input Metrics", T.StructType([T.StructField("Bytes Read", L), T.StructField("Records Read", L)])),
    T.StructField("Output Metrics", T.StructType([T.StructField("Bytes Written", L), T.StructField("Records Written", L)])),
])
EVENT_SCHEMA = T.StructType([
    T.StructField("Event", T.StringType()), T.StructField("Stage ID", L),
    T.StructField("Task Info", T.StructType([
        T.StructField("Task ID", L), T.StructField("Failed", T.BooleanType()), T.StructField("Killed", T.BooleanType()),
        T.StructField("Accumulables", T.ArrayType(T.StructType([
            T.StructField("Name", T.StringType()), T.StructField("Update", T.StringType())])))])),
    T.StructField("Task Metrics", TM_SCHEMA),
])


def metric_columns():
    """Column expressions for the 24 per-task metrics, in TASK_FEATURES order."""
    cols = []
    for _, path in TASK_FEATURES[:N_TASK_METRICS]:
        if path[0] == "acc":
            acc = F.filter(F.col("e.`Task Info`.Accumulables"), lambda a: a["Name"] == F.lit(path[1]))
            c = F.element_at(acc, 1)["Update"].cast("double")
        else:
            c = F.col("e." + ".".join(f"`{p}`" for p in path))
        cols.append(F.coalesce(c.cast("double"), F.lit(0.0)))
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--logs", default=str(ROOT / "data" / "raw" / "eventlogs"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "spark"))
    ap.add_argument("--partitions", type=int, default=0, help="shuffle partitions (0 = 4 x cores)")
    ap.add_argument("--driver-memory", default="4g")
    ap.add_argument("--tag", default="")
    ap.add_argument("--glob", default="*/*/*", help="subset of logs, e.g. 'large/big_executor3_cores3_memory5g_shuffle200/*'")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # scratch space for shuffles/spills; on mounted folders Spark may not create its directories -> use local disk
    tmp = Path(os.environ.get("AI508_SPARK_TMP", str(ROOT / "data" / "tmp" / "spark-local")))
    tmp.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    b = (SparkSession.builder.master(a.master).appName("garralda-featuriser")
         .config("spark.driver.memory", a.driver_memory)
         .config("spark.local.dir", str(tmp))
         .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
         .config("spark.ui.showConsoleProgress", "false"))
    if not a.master.startswith("local"):
        b = b.config("spark.eventLog.enabled", "false")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    cores = spark.sparkContext.defaultParallelism
    spark.conf.set("spark.sql.shuffle.partitions", str(a.partitions or max(8, 4 * cores)))
    timings = {"session_s": round(time.time() - t_start, 2)}

    # ---- 1. read and parse ------------------------------------------------------------------------------
    t0 = time.time()
    text = spark.read.text(f"{a.logs}/{a.glob}").withColumn("app", F.element_at(F.split(F.input_file_name(), "/"), -1))
    # cheap prefix filters (Spark writes "Event" as the first key); JSON is parsed only for the kept lines
    task_lines = text.where(F.col("value").startswith('{"Event":"SparkListenerTaskEnd"'))
    events = (text.where(F.substring("value", 1, 64).rlike(
        '^[{]"Event":"SparkListener(ApplicationStart|ApplicationEnd|EnvironmentUpdate|LogStart)"'))
        .withColumn("ev", F.get_json_object("value", "$.Event")))
    tasks = (task_lines.select("app", F.from_json("value", EVENT_SCHEMA).alias("e"))
             .where(F.col("e.`Task Metrics`").isNotNull())
             .select("app", F.col("e.`Stage ID`").alias("stage"), F.col("e.`Task Info`.`Task ID`").alias("task_id"),
                     F.array(*metric_columns()).alias("vals")))
    tasks = tasks.persist()
    n_tasks = tasks.count()
    timings["parse_tasks_s"] = round(time.time() - t0, 2)

    # application metadata (small)
    t0 = time.time()
    meta_ev = events
    props = F.from_json(F.get_json_object("value", "$['Spark Properties']"), T.MapType(T.StringType(), T.StringType()))
    meta = (meta_ev.groupBy("app").agg(
        F.max(F.when(F.col("ev") == "SparkListenerApplicationStart", F.get_json_object("value", "$.Timestamp").cast("long"))).alias("t_start"),
        F.max(F.when(F.col("ev") == "SparkListenerApplicationEnd", F.get_json_object("value", "$.Timestamp").cast("long"))).alias("t_end"),
        F.max(F.when(F.col("ev") == "SparkListenerLogStart", F.get_json_object("value", "$['Spark Version']"))).alias("spark_version"),
        F.first(F.when(F.col("ev") == "SparkListenerEnvironmentUpdate", props), ignorenulls=True).alias("props"))
        .select("app", ((F.col("t_end") - F.col("t_start")) / 1000.0).alias("exec_time_s"), "spark_version",
                F.coalesce(F.col("props")["spark.driver.cores"], F.lit("1")).cast("int").alias("driver_cores"),
                F.col("props")["spark.driver.memory"].alias("driver_memory"),
                F.col("props")["spark.executor.cores"].cast("int").alias("executor_cores"),
                F.col("props")["spark.executor.instances"].cast("int").alias("executor_instances"),
                F.col("props")["spark.executor.memory"].alias("executor_memory"),
                F.coalesce(F.col("props")["spark.sql.shuffle.partitions"], F.lit("200")).cast("int").alias("sql_shuffle_partitions")))
    meta_pd = meta.toPandas()
    timings["metadata_s"] = round(time.time() - t0, 2)

    # ---- 2. long format ---------------------------------------------------------------------------------
    t0 = time.time()
    long_tasks = tasks.select("app", F.col("task_id").cast("double").alias("ord"),
                              F.posexplode("vals").alias("m", "v"))
    per_stage = (tasks.groupBy("app", "stage").count()
                 .select("app", F.col("stage").cast("double").alias("ord"), F.lit(N_TASK_METRICS).alias("m"),
                         F.col("count").cast("double").alias("v")))
    long = long_tasks.unionByName(per_stage)

    # ---- 3. window transformations (JPDC Eqs. 5-7 + final min-max) ----------------------------------------
    g = Window.partitionBy("app", "m")
    go = g.orderBy("ord")
    s = (long.withColumn("n", F.count("v").over(g))
         .withColumn("vmin", F.min("v").over(g)).withColumn("vmax", F.max("v").over(g))
         .where(F.col("vmax") > F.col("vmin"))                                   # constant series -> zeros
         .withColumn("t", (F.col("v") - F.col("vmin")) / (F.col("vmax") - F.col("vmin")))    # Eq. 5
         .withColumn("z", (F.col("t") - F.avg("t").over(g)) / F.stddev_pop("t").over(g))      # Eq. 6
         .withColumn("cus", F.sum("z").over(go.rowsBetween(Window.unboundedPreceding, Window.currentRow)))  # Eq. 7
         .withColumn("cmin", F.min("cus").over(g)).withColumn("cmax", F.max("cus").over(g))
         .withColumn("y", (F.col("cus") - F.col("cmin")) / (F.col("cmax") - F.col("cmin")))
         .withColumn("x", F.when(F.col("n") > 1, (F.row_number().over(go) - 1) / (F.col("n") - 1)).otherwise(0.0))
         .select("app", "m", "n", "x", "y"))
    s = s.persist()

    # ---- 4. quadratic least squares from moments ----------------------------------------------------------
    mom = s.groupBy("app", "m").agg(
        F.first("n").alias("n"), F.sum("x").alias("s1"), F.sum(F.pow("x", 2)).alias("s2"),
        F.sum(F.pow("x", 3)).alias("s3"), F.sum(F.pow("x", 4)).alias("s4"), F.sum("y").alias("t0"),
        F.sum(F.col("x") * F.col("y")).alias("t1"), F.sum(F.pow("x", 2) * F.col("y")).alias("t2")).toPandas()
    small_groups = mom[mom.n <= 3][["app", "m"]]
    coef = {}
    for r in mom.itertuples():
        if r.n > 3:
            A = np.array([[r.s4, r.s3, r.s2], [r.s3, r.s2, r.s1], [r.s2, r.s1, r.n]], float)
            coef[(r.app, r.m)] = np.linalg.solve(A, np.array([r.t2, r.t1, r.t0], float))
    if len(small_groups):
        keys = spark.createDataFrame(small_groups)
        pts = s.join(F.broadcast(keys), ["app", "m"]).toPandas()
        for (app, m), grp in pts.groupby(["app", "m"]):
            grp = grp.sort_values("x")
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coef[(app, m)] = np.polyfit(grp.x.values, grp.y.values, 2)
    import pandas as pd
    cdf = pd.DataFrame([(k[0], int(k[1]), *v) for k, v in coef.items()], columns=["app", "m", "a", "b", "c"])
    cs = spark.createDataFrame(cdf)
    mae = (s.join(F.broadcast(cs), ["app", "m"])
           .groupBy("app", "m").agg(F.avg(F.abs(F.col("y") - (F.col("a") * F.col("x") * F.col("x") + F.col("b") * F.col("x") + F.col("c")))).alias("mae"))
           .toPandas())
    timings["descriptor_s"] = round(time.time() - t0, 2)

    # ---- 5. Prats-descriptor -------------------------------------------------------------------------------
    t0 = time.time()
    prats = (long.groupBy("app", "m").agg(F.avg("v").alias("mean"), F.min("v").alias("min"), F.max("v").alias("max"),
                                          F.stddev_pop("v").alias("std")).toPandas())
    sums = tasks.groupBy("app").agg(F.count("*").alias("n_tasks"),
                                    F.sum(F.element_at("vals", 12)).alias("input_bytes")).toPandas()
    timings["prats_s"] = round(time.time() - t0, 2)

    # ---- assemble ------------------------------------------------------------------------------------------
    apps = sorted(meta_pd.app)
    G = {a_: np.zeros(100) for a_ in apps}
    full = cdf.merge(mae, on=["app", "m"])
    for r in full.itertuples():
        G[r.app][4 * r.m:4 * r.m + 4] = [r.a, r.b, r.c, r.mae]
    P = {a_: np.zeros(100) for a_ in apps}
    for r in prats.itertuples():
        P[r.app][4 * r.m:4 * r.m + 4] = [r.mean, r.min, r.max, 0.0 if r.std != r.std else r.std]
    from src.featuriser.descriptor import descriptor_columns
    pd.DataFrame([[a_] + list(G[a_]) for a_ in apps], columns=["app"] + descriptor_columns("g")).to_csv(out / "garralda_spark.csv", index=False, float_format="%.17g")
    pd.DataFrame([[a_] + list(P[a_]) for a_ in apps], columns=["app"] + descriptor_columns("p")).to_csv(out / "prats_spark.csv", index=False, float_format="%.17g")
    meta_pd.merge(sums, on="app").to_csv(out / "apps_spark.csv", index=False)
    timings["total_s"] = round(time.time() - t_start, 2)
    info = {"master": a.master, "default_parallelism": cores, "shuffle_partitions": int(spark.conf.get("spark.sql.shuffle.partitions")),
            "apps": len(apps), "tasks": int(n_tasks), "long_rows_task_metrics": int(n_tasks * N_TASK_METRICS),
            "spark_version": spark.version, "timings": timings, "tag": a.tag,
            "log_bytes": sum(p.stat().st_size for p in Path(a.logs).glob(a.glob) if p.is_file()), "glob": a.glob}
    (out / f"run_{a.tag or a.master.replace('[', '_').replace(']', '').replace('*', 'all').replace(':', '_').replace('/', '')}.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
