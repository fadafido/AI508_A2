#!/usr/bin/env bash
# Builds the two HiBench jars the 2022 runs used: HiBench master (8.0-SNAPSHOT) at bf390d2, the last
# commit before the runs (Jan 2022), sparkbench profile for Spark 2.4 / Scala 2.11.
# Needs git, Maven 3 and Java 8. The jars end up in tools/hibench/ of the project.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
SRC=$WORK/src/HiBench
[ -d "$SRC" ] || git clone -q https://github.com/Intel-bigdata/HiBench.git "$SRC"
cd "$SRC"
git checkout -q bf390d2e60ed05f77264820b79f4637910e9629f
mvn -B -q -Psparkbench -Dspark=2.4 -Dscala=2.11 -Dmaven.javadoc.skip=true -DskipTests clean package
mkdir -p "$PROJECT/tools/hibench"
cp sparkbench/assembly/target/sparkbench-assembly-8.0-SNAPSHOT-dist.jar \
   autogen/target/autogen-8.0-SNAPSHOT-jar-with-dependencies.jar "$PROJECT/tools/hibench/"
git rev-parse HEAD > "$PROJECT/tools/hibench/COMMIT"
ls -la "$PROJECT/tools/hibench"
