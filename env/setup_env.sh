#!/usr/bin/env bash
# Create the two Python environments (outside the project folder) and check Java.
#   VENV      main environment       (default ~/.venvs/ai508a2)
#   VENV26    FGCS 2026 anchor env   (default ~/.venvs/ai508a2-fgcs2026)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python3}
VENV=${VENV:-$HOME/.venvs/ai508a2}
VENV26=${VENV26:-$HOME/.venvs/ai508a2-fgcs2026}
$PYTHON -m venv "$VENV" && "$VENV/bin/pip" install -q --upgrade pip && "$VENV/bin/pip" install -q -r env/requirements.txt
$PYTHON -m venv "$VENV26" && "$VENV26/bin/pip" install -q --upgrade pip && "$VENV26/bin/pip" install -q -r env/requirements-fgcs2026.txt
java -version 2>&1 | head -1 || echo "Java 11 or 17 is needed for PySpark 3.5"
"$VENV/bin/python" -c "import numpy, scipy, sklearn, pandas, pyspark; print('main env:', numpy.__version__, scipy.__version__, sklearn.__version__, pandas.__version__, pyspark.__version__)"
"$VENV26/bin/python" -c "import numpy, scipy, sklearn; print('fgcs2026 env:', numpy.__version__, scipy.__version__, sklearn.__version__)"
