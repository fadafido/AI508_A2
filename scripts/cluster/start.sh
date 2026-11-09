#!/usr/bin/env bash
# Starts a small cluster on this machine:
#   HDFS  - NameNode + DataNode on hdfs://127.0.0.1:9000, replication 1
#   Spark - standalone master + $N_WORKERS workers, event logging on
#   MapReduce for the HiBench data generators runs in local mode (no YARN)
# The 2022 cluster used YARN (client mode) with 3 worker containers. Without YARN/Docker the closest
# thing is Spark standalone with 3 worker daemons.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
mkdir -p "$HADOOP_CONF_DIR" "$SPARK_CONF_DIR" "$HADOOP_LOG_DIR" "$SPARK_LOG_DIR" "$WORK/pids" "$EVENTLOG_TMP" \
         "$WORK/hdfs/name" "$WORK/hdfs/data" "$WORK/tmp" "$WORK/spark-work" "$WORK/spark-local"

# fresh copy of the default config every time, then our changes on top
cp "$HADOOP_HOME"/etc/hadoop/* "$HADOOP_CONF_DIR"/
cat > "$HADOOP_CONF_DIR/core-site.xml" <<EOF
<configuration>
  <property><name>fs.defaultFS</name><value>$NN</value></property>
  <property><name>hadoop.tmp.dir</name><value>$WORK/tmp</value></property>
</configuration>
EOF
cat > "$HADOOP_CONF_DIR/hdfs-site.xml" <<EOF
<configuration>
  <property><name>dfs.replication</name><value>1</value></property>
  <property><name>dfs.namenode.name.dir</name><value>file://$WORK/hdfs/name</value></property>
  <property><name>dfs.datanode.data.dir</name><value>file://$WORK/hdfs/data</value></property>
  <property><name>dfs.permissions.enabled</name><value>false</value></property>
  <property><name>dfs.namenode.datanode.registration.ip-hostname-check</name><value>false</value></property>
  <property><name>dfs.datanode.address</name><value>127.0.0.1:50010</value></property>
  <property><name>dfs.datanode.http.address</name><value>127.0.0.1:50075</value></property>
  <property><name>dfs.datanode.ipc.address</name><value>127.0.0.1:50020</value></property>
  <property><name>dfs.namenode.http-address</name><value>127.0.0.1:50070</value></property>
</configuration>
EOF
cat > "$HADOOP_CONF_DIR/mapred-site.xml" <<EOF
<configuration>
  <property><name>mapreduce.framework.name</name><value>local</value></property>
  <property><name>mapreduce.cluster.local.dir</name><value>$WORK/tmp/mapred</value></property>
</configuration>
EOF
# small daemon heaps, the VM has ~4 GB in total. Client commands (data generation) get more.
cat >> "$HADOOP_CONF_DIR/hadoop-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME
export HADOOP_NAMENODE_OPTS="-Xmx256m \$HADOOP_NAMENODE_OPTS"
export HADOOP_DATANODE_OPTS="-Xmx256m \$HADOOP_DATANODE_OPTS"
export HADOOP_CLIENT_OPTS="-Xmx768m"
EOF
cat > "$SPARK_CONF_DIR/spark-env.sh" <<EOF
export JAVA_HOME=$JAVA_HOME
export HADOOP_CONF_DIR=$HADOOP_CONF_DIR
export SPARK_MASTER_HOST=127.0.0.1
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_LOCAL_HOSTNAME=localhost
export SPARK_DAEMON_MEMORY=256m
export SPARK_WORKER_INSTANCES=$N_WORKERS
export SPARK_WORKER_CORES=$WORKER_CORES
export SPARK_WORKER_MEMORY=$WORKER_MEM
export SPARK_WORKER_DIR=$WORK/spark-work
export SPARK_LOCAL_DIRS=$WORK/spark-local
export SPARK_LOG_DIR=$SPARK_LOG_DIR
export SPARK_PID_DIR=$SPARK_PID_DIR
EOF
cat > "$SPARK_CONF_DIR/spark-defaults.conf" <<EOF
spark.master                      $SPARK_MASTER_URL
spark.eventLog.enabled            true
spark.eventLog.dir                file://$EVENTLOG_TMP
spark.driver.host                 127.0.0.1
spark.dynamicAllocation.enabled   false
spark.local.dir                   $WORK/spark-local
EOF

if [ ! -f "$WORK/hdfs/name/current/VERSION" ]; then
  hdfs namenode -format -force -nonInteractive > "$HADOOP_LOG_DIR/format.log" 2>&1
fi
hadoop-daemon.sh start namenode
hadoop-daemon.sh start datanode
for i in $(seq 1 40); do
  hdfs dfsadmin -report 2>/dev/null | grep -q "Live datanodes (1)" && break
  sleep 1
done
hdfs dfsadmin -safemode leave >/dev/null 2>&1 || true

start-master.sh >/dev/null
for i in $(seq 1 20); do curl -s http://127.0.0.1:8080 >/dev/null && break; sleep 1; done
start-slave.sh "$SPARK_MASTER_URL" >/dev/null
for i in $(seq 1 30); do
  n=$(curl -s http://127.0.0.1:8080/json/ | grep -o '"state" : "ALIVE"' | wc -l)
  [ "$n" -ge "$N_WORKERS" ] && break
  sleep 1
done
echo "hdfs: $(hdfs dfsadmin -report 2>/dev/null | grep 'Live datanodes')  spark workers alive: $n"
