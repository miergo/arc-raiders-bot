#!/usr/bin/env bash
# Start the MLflow tracking server with a local file store.
# Traces and metrics are persisted under ./mlruns/ by default.
#
# Usage:
#   ./run_mlflow_ui.sh            # default port 5000
#   ./run_mlflow_ui.sh 5001       # custom port

set -euo pipefail

PORT="${1:-5000}"

echo "Starting MLflow UI on http://localhost:${PORT} ..."
exec mlflow server --host 0.0.0.0 --port "$PORT"
