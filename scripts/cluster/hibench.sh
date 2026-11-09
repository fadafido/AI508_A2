#!/usr/bin/env bash
# Generate input data for / run one HiBench workload on the local cluster. This does what HiBench's
# bin/workloads/*/prepare.sh and spark/run.sh scripts do (HiBench master @ bf390d2, same as 2022) but
# without its config layer (python2 scripts that expect YARN).
#
#   scripts/cluster/hibench.sh prepare <workload> <profile>
#   scripts/cluster/hibench.sh run <workload> <profile> <setting> [rep]
#
# workloads: sort terasort wordcount repartition kmeans pagerank bayes sleep
# profiles:  tiny (called "small" in the paper) and small ("medium"), see spark.app.benchmark.data.size
#            in the original logs
# settings:  1 = 1 executor x 4 cores, 4g, 200 shuffle partitions
#            2 = 3 x 3 cores, 3g, 200
#            3 = 3 x 3 cores, 3g, 100
#            (0 = 1 x 1 core, 1g, 8 -- quick smoke test)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
# Local-mode MapReduce (the HiBench data generators) creates its DistributedCache symlinks in the
# current directory, and the shared project mount refuses symlinks. Everything below uses absolute
# paths, so run from the work folder on the VM's own disk.
cd "$WORK"
action=$1; W=$2; P=$3; S=${4:-2}; REP=${5:-1}

MAPS=2      # hibench.default.map.parallelism, the original runs read 2 input files
case $S in
  0) EI=1; EC=1; EM_GB=1; SH=8 ;;
  1) EI=1; EC=4; EM_GB=4; SH=200 ;;
  2) EI=3; EC=3; EM_GB=3; SH=200 ;;
  3) EI=3; EC=3; EM_GB=3; SH=100 ;;
  *) echo "unknown setting $S"; exit 1 ;;
esac
EM=$(( EM_GB * 1024 / MEM_DIV ))m
case $P in tiny) SCALE=small ;; small) SCALE=medium ;; *) echo "unknown profile $P"; exit 1 ;; esac

declare -A NAME=([sort]=Sort [terasort]=Terasort [wordcount]=Wordcount [repartition]=Repartition
                 [kmeans]=Kmeans [pagerank]=Pagerank [bayes]=Bayes [sleep]=Sleep)
BASE=$NN/HiBench
IN=$BASE/${NAME[$W]}/Input
OUTPUT=$BASE/${NAME[$W]}/Output
mkdir -p "$OUT/eventlogs" "$OUT/driver_logs" "$WORK/props"

# HiBench reads hibench.default.shuffle.parallelism etc. from this file
pfile=$WORK/props/sparkbench_s$S.properties
cat > "$pfile" <<EOF
hibench.default.map.parallelism     $MAPS
hibench.default.shuffle.parallelism $SH
sparkbench.inputformat              Sequence
sparkbench.outputformat             Sequence
EOF

# data sizes from conf/workloads/*.conf
datasize() {
  case "$W:$P" in
    sort:tiny|wordcount:tiny|terasort:tiny) echo 32000 ;;
    sort:small|terasort:small) echo 3200000 ;;
    wordcount:small) echo 320000000 ;;
    repartition:tiny) echo 3200 ;;
    repartition:small) echo 3200000 ;;
  esac
}

now() { date +%s.%N; }
secs() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", b-a}'; }

if [ "$action" = prepare ]; then
  t0=$(now)
  hadoop fs -rm -r -f -skipTrash "$IN" "$IN.parquet" >/dev/null 2>&1
  case $W in
    sort|wordcount)
      D=$(datasize)
      hadoop jar "$HADOOP_EXAMPLES_JAR" randomtextwriter -D mapreduce.randomtextwriter.totalbytes=$D \
        -D mapreduce.randomtextwriter.bytespermap=$((D / MAPS)) -D mapreduce.job.maps=$MAPS \
        -D mapreduce.job.reduces=$MAPS "$IN" ;;
    terasort|repartition)
      hadoop jar "$HADOOP_EXAMPLES_JAR" teragen -D mapreduce.job.maps=$MAPS -D mapreduce.job.reduces=$MAPS \
        "$(datasize)" "$IN" ;;
    kmeans)
      if [ "$P" = tiny ]; then N=30000; F=6000; DIM=3; else N=3000000; F=600000; DIM=20; fi
      hadoop jar "$AUTOGEN_JAR" org.apache.mahout.clustering.kmeans.GenKMeansDataset \
        -D hadoop.job.history.user.location="$IN/samples" -sampleDir "$IN/samples" -clusterDir "$IN/cluster" \
        -numClusters 5 -numSamples $N -samplesPerFile $F -sampleDimension $DIM ;;
    pagerank)
      if [ "$P" = tiny ]; then PG=50; else PG=5000; fi
      hadoop jar "$AUTOGEN_JAR" HiBench.DataGen -t pagerank -b "$BASE/Pagerank" -n Input -m $MAPS -r $MAPS \
        -p $PG -pbalance -pbalance -o text ;;
    bayes)
      if [ "$P" = tiny ]; then PG=25000; CL=10; EX=100; else PG=30000; CL=100; EX=1000; fi
      hadoop jar "$AUTOGEN_JAR" HiBench.DataGen -t bayes -b "$BASE/Bayes" -n Input -m $MAPS -r $MAPS \
        -p $PG -class $CL -o sequence &&
      SPARKBENCH_PROPERTIES_FILES=$pfile spark-submit --master "$SPARK_MASTER_URL" --conf spark.eventLog.enabled=false \
        --driver-memory "$DRIVER_MEM" --executor-memory 512m --class HiBench.BayesDataGen "$AUTOGEN_JAR" \
        --input "$IN" --output "$IN.parquet" --useDense false --examples $EX --features 100 --classes $CL ;;
    sleep) ;;   # no input
  esac
  rc=$?
  size=$(hadoop fs -du -s "$IN" 2>/dev/null | awk '{print $1}')
  echo "{\"workload\":\"$W\",\"profile\":\"$P\",\"rc\":$rc,\"wall_s\":$(secs "$t0" "$(now)"),\"input_bytes\":${size:-0}}" \
    | tee -a "$OUT/prepare.jsonl"
  exit $rc
fi

case $W in
  sort)        CLS=com.intel.hibench.sparkbench.micro.ScalaSort; ARGS="$IN $OUTPUT" ;;
  terasort)    CLS=com.intel.hibench.sparkbench.micro.ScalaTeraSort; ARGS="$IN $OUTPUT" ;;
  wordcount)   CLS=com.intel.hibench.sparkbench.micro.ScalaWordCount; ARGS="$IN $OUTPUT" ;;
  repartition) CLS=com.intel.hibench.sparkbench.micro.ScalaRepartition; ARGS="$IN $OUTPUT true true" ;;
  kmeans)      CLS=com.intel.hibench.sparkbench.ml.DenseKMeans
               ARGS="-k 10 --numIterations 5 --storageLevel MEMORY_ONLY --initMode Random $IN/samples" ;;
  pagerank)    CLS=org.apache.spark.examples.SparkPageRank
               if [ "$P" = tiny ]; then IT=1; else IT=3; fi; ARGS="$IN/edges $OUTPUT $IT" ;;
  bayes)       CLS=com.intel.hibench.sparkbench.ml.NaiveBayesExample; ARGS="$IN.parquet" ;;
  sleep)       CLS=com.intel.hibench.sparkbench.micro.ScalaSleep
               if [ "$P" = tiny ]; then ARGS=3; else ARGS=30; fi ;;
esac
hadoop fs -rm -r -f -skipTrash "$OUTPUT" >/dev/null 2>&1
# minRegisteredResourcesRatio: wait for the executors before the first job like YARN does (0.8 is the YARN
# default, standalone uses 0). Without it sc.defaultParallelism is still 2 when ScalaSleep starts and it
# runs 2 tasks instead of one per core like in the 2022 logs.
tag=${W}_${P}_s${S}_r${REP}
log=$WORK/driver_$tag.out
started=$(date -u +%FT%TZ)
t0=$(now)
SPARKBENCH_PROPERTIES_FILES=$pfile spark-submit --master "$SPARK_MASTER_URL" --deploy-mode client --class $CLS \
  --driver-memory "$DRIVER_MEM" --executor-cores $EC --executor-memory $EM --total-executor-cores $((EI * EC)) \
  --conf spark.executor.instances=$EI --conf spark.sql.shuffle.partitions=$SH \
  --conf spark.scheduler.minRegisteredResourcesRatio=0.8 \
  --conf spark.app.benchmark.workload=$W --conf spark.app.benchmark.data.size=$P \
  --conf spark.app.benchmark.group.id=$CAMPAIGN --conf spark.app.benchmark.setting=$S \
  --conf spark.app.benchmark.rep=$REP \
  "$HIBENCH_JAR" $ARGS > "$log" 2>&1
rc=$?
wall=$(secs "$t0" "$(now)")

app=$(grep -o 'app-[0-9]*-[0-9]*' "$log" | head -1)
if [ -n "$app" ] && [ -f "$EVENTLOG_TMP/$app" ]; then
  # failed runs are kept apart so they don't end up in the dataset
  [ $rc -eq 0 ] && d=$OUT/eventlogs/$SCALE/setting$S || d=$OUT/eventlogs_failed
  mkdir -p "$d"
  cp "$EVENTLOG_TMP/$app" "$d/${app}_$W"
fi
# each executor gets its own copy of the 118 MB HiBench jar in the worker dir, the VM disk fills up fast
[ -n "$app" ] && rm -rf "$WORK/spark-work/$app"
gzip -c "$log" > "$OUT/driver_logs/$tag.out.gz" && rm -f "$log"
echo "{\"tag\":\"$tag\",\"campaign\":\"$CAMPAIGN\",\"workload\":\"$W\",\"profile\":\"$P\",\"scale\":\"$SCALE\",\"setting\":$S,\"rep\":$REP,\"app\":\"$app\",\"rc\":$rc,\"wall_s\":$wall,\"started\":\"$started\",\"executor_memory\":\"$EM\",\"driver_memory\":\"$DRIVER_MEM\",\"workers\":$N_WORKERS}" \
  | tee -a "$OUT/runs.jsonl"
exit $rc
