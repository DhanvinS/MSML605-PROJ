#!/usr/bin/env bash
# Train XGBoost models locally using data from data/processed/.
# Output goes to models/latest/ and output/ instead of SageMaker paths.
#
# Usage:
#   bash scripts/train_local.sh
#
# Run ingestion first:
#   bash scripts/ingest_local.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="$PROJECT_ROOT/data/processed"
MODEL_DIR="$PROJECT_ROOT/models/latest"
OUTPUT_DIR="$PROJECT_ROOT/output"

if [[ -z "$(ls "$DATA_DIR"/*.parquet 2>/dev/null)" ]]; then
    echo "ERROR: No parquet files found in $DATA_DIR"
    echo "Run ingestion first: bash scripts/ingest_local.sh"
    exit 1
fi

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

export SM_CHANNEL_TRAIN="$DATA_DIR"
export SM_MODEL_DIR="$MODEL_DIR"
export SM_OUTPUT_DATA_DIR="$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT"

echo "Training XGBoost models..."
echo "  Data dir  : $DATA_DIR"
echo "  Model dir : $MODEL_DIR"
echo "  Output    : $OUTPUT_DIR"
echo ""

cd "$PROJECT_ROOT"
python src/training/train_xgboost.py

echo ""
echo "Done. Models in $MODEL_DIR:"
ls -lh "$MODEL_DIR"
echo ""
echo "Metrics: $OUTPUT_DIR/metrics.json"
