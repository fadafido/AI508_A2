#!/usr/bin/env bash
# Record the computing environment of this machine in results/env/<label>.txt
set -uo pipefail
cd "$(dirname "$0")/.."
LABEL=${1:-$(hostname)}
PY=${PY:-python3}
mkdir -p results/env
{
  echo "label: $LABEL"; echo "date_utc: $(date -u '+%F %T')"
  echo "kernel: $(uname -srm)"
  echo "cpus: $(nproc 2>/dev/null || sysctl -n hw.ncpu)"
  echo "cpu: $(grep -m1 -i 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || sysctl -n machdep.cpu.brand_string)"
  echo "cpu_implementer: $(grep -m1 'CPU implementer' /proc/cpuinfo 2>/dev/null | cut -d: -f2)"
  echo "memory: $(free -h 2>/dev/null | awk '/Mem:/{print $2}')"
  echo "os: $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2)"
  echo "disk_project: $(df -h . | awk 'NR==2 {print $2 " total, " $4 " free"}')"
  echo "java: $(java -version 2>&1 | grep -v "Picked up" | head -1)"
  echo "python: $($PY --version 2>&1)"
  $PY - <<'PY'
import importlib
for m in ["numpy", "scipy", "sklearn", "pandas", "joblib", "matplotlib", "pyspark"]:
    try:
        print(f"{m}: {importlib.import_module(m).__version__}")
    except Exception as e:
        print(f"{m}: n/a")
PY
} > "results/env/$LABEL.txt"
cat "results/env/$LABEL.txt"
