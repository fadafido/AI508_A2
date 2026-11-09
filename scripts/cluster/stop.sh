#!/usr/bin/env bash
# Stops everything start.sh started.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
SPARK_WORKER_INSTANCES=$N_WORKERS stop-slave.sh >/dev/null 2>&1
stop-master.sh >/dev/null 2>&1
hadoop-daemon.sh stop datanode >/dev/null 2>&1
hadoop-daemon.sh stop namenode >/dev/null 2>&1
# leftovers (e.g. after a killed session)
pkill -f org.apache.spark.deploy.worker.Worker 2>/dev/null
pkill -f org.apache.spark.deploy.master.Master 2>/dev/null
pkill -f org.apache.hadoop.hdfs.server 2>/dev/null
rm -f "$WORK"/pids/*.pid
echo stopped
