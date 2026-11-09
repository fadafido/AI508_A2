# Shared settings for the cluster scripts. Source it, don't run it.
#
# The big stuff (JDK, Spark, Hadoop, HDFS blocks, scratch dirs) goes under $WORK on a local disk.
# The project folder is a shared mount on my setup and HDFS/Spark fail there when they try to
# chmod their dirs, so only the finished event logs and run records are copied into data/cluster/.

PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK=${AI508_WORK:-$HOME/ai508-cluster}
TOOLS=${TOOLS:-$WORK/tools}

export JAVA_HOME=${JAVA8_HOME:-$(ls -d "$TOOLS"/jdk8* 2>/dev/null | head -1)}
export HADOOP_HOME=$TOOLS/hadoop-2.7.7
export SPARK_HOME=$TOOLS/spark-2.4.8-bin-hadoop2.7
export HADOOP_CONF_DIR=$WORK/conf/hadoop
export SPARK_CONF_DIR=$WORK/conf/spark
export HADOOP_LOG_DIR=$WORK/logs/hadoop
export SPARK_LOG_DIR=$WORK/logs/spark
export HADOOP_PID_DIR=$WORK/pids
export SPARK_PID_DIR=$WORK/pids
export PATH=$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

HIBENCH_JAR=$TOOLS/hibench/sparkbench-assembly-8.0-SNAPSHOT-dist.jar
AUTOGEN_JAR=$TOOLS/hibench/autogen-8.0-SNAPSHOT-jar-with-dependencies.jar
HADOOP_EXAMPLES_JAR=$HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.7.jar

NN=hdfs://127.0.0.1:9000
SPARK_MASTER_URL=spark://127.0.0.1:7077
EVENTLOG_TMP=$WORK/eventlogs        # Spark writes the logs here first
OUT=${AI508_OUT:-$PROJECT/data/cluster}   # copied here after each run (eventlogs/, runs.jsonl, driver output)
CAMPAIGN=${AI508_CAMPAIGN:-rerun-vm}      # benchmark group id, so campaigns can't mix

# 3 workers like the 2022 cluster. Cores are oversubscribed (the VM has 4) and executor memory
# is divided by MEM_DIV in hibench.sh (see docs/cluster.md).
N_WORKERS=${N_WORKERS:-3}
WORKER_CORES=${WORKER_CORES:-4}
WORKER_MEM=${WORKER_MEM:-1g}
MEM_DIV=${MEM_DIV:-4}
DRIVER_MEM=${DRIVER_MEM:-512m}
