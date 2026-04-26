#!/usr/bin/env bash
# Start the stock prediction inference server locally using models/latest/.
# The server is the same FastAPI app deployed on AWS SageMaker.
#
# Usage:
#   bash scripts/run_local.sh
#   MODEL_DIR=/path/to/models bash scripts/run_local.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models/latest}"

# Verify model files exist before starting
if [[ ! -f "$MODEL_DIR/model.xgb" ]]; then
    echo "ERROR: model.xgb not found in $MODEL_DIR"
    echo "Run the training pipeline first: bash scripts/04_initial_train.sh"
    exit 1
fi

export MODEL_DIR
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting inference server..."
echo "  Model dir : $MODEL_DIR"
echo "  URL       : http://127.0.0.1:8000"
echo "  Health    : http://127.0.0.1:8000/health"
echo "  Docs      : http://127.0.0.1:8000/docs"
echo ""

cd "$PROJECT_ROOT"
uvicorn src.inference.app:app --host 127.0.0.1 --port 8000 --log-level info
