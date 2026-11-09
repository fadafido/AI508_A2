#!/usr/bin/env bash
# Clone the authors' repositories at the pinned commits into external/.
set -euo pipefail
cd "$(dirname "$0")/.."
clone() { # url dir commit
  if [ ! -d "external/$2/.git" ] && [ ! -f "external/$2/README.md" ]; then git clone -q "$1" "external/$2"; fi
  if [ -d "external/$2/.git" ]; then (cd "external/$2" && git fetch -q --depth 1 origin "$3" 2>/dev/null || true; git checkout -q "$3"); fi
  echo "$2 @ $3"
}
clone https://github.com/mgarralda/garralda-performance-model.git garralda-performance-model 9f23b627c6afa3e753542f2d6f2aea61b0f59009
clone https://github.com/mgarralda/spark-self-tuning-framework.git spark-self-tuning-framework 811b780a385db793d622ddaaed4e87769154f3fd
