#!/usr/bin/env bash
# Downloads and unpacks JDK 8, Spark 2.4.8 and Hadoop 2.7.7 into $TOOLS (same versions as the 2022 runs).
# Downloads resume if the script gets interrupted, just run it again.
# HiBench jars: build them with build_hibench.sh, or drop the two jars in tools/hibench/ of the project.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
mkdir -p "$TOOLS/dl" "$TOOLS/hibench"

case $(uname -m) in arm64|aarch64) ARCH=aarch64 ;; *) ARCH=x64 ;; esac
case $(uname -s) in Darwin) OS=mac ;; *) OS=linux ;; esac

get() {
  local f=$TOOLS/dl/$2
  [ -s "$f" ] && return 0
  echo "downloading $2"
  curl -fsSL --retry 3 -C - -o "$f.part" "$1" && mv "$f.part" "$f"
}
get "https://api.adoptium.net/v3/binary/latest/8/ga/$OS/$ARCH/jdk/hotspot/normal/eclipse?project=jdk" "jdk8-$OS-$ARCH.tgz"
get https://archive.apache.org/dist/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz spark-2.4.8-bin-hadoop2.7.tgz
get https://archive.apache.org/dist/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz hadoop-2.7.7.tar.gz

cd "$TOOLS"
ls -d jdk8* >/dev/null 2>&1 || tar xzf "dl/jdk8-$OS-$ARCH.tgz"
[ -d spark-2.4.8-bin-hadoop2.7 ] || tar xzf dl/spark-2.4.8-bin-hadoop2.7.tgz
[ -d hadoop-2.7.7 ] || tar xzf dl/hadoop-2.7.7.tar.gz

# HiBench jars kept in the project (not in git, too big)
for j in "$PROJECT"/tools/hibench/*.jar; do
  [ -f "$j" ] && [ ! -s "$TOOLS/hibench/$(basename "$j")" ] && cp "$j" "$TOOLS/hibench/"
done

source "$HERE/env.sh"
java -version 2>&1 | head -1
ls "$TOOLS" "$TOOLS/hibench"
