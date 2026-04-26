#!/usr/bin/env bash
# Fetch OHLCV data and build feature matrices locally.
# Output goes to data/processed/ instead of SageMaker's /opt/ml/processing/output/.
#
# Usage:
#   bash scripts/ingest_local.sh
#   TICKERS="AAPL MSFT" LOOKBACK_DAYS=365 bash scripts/ingest_local.sh
#
# Requires defeatbeta-api:
#   pip install -r requirements-ingestion.txt

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TICKERS="${TICKERS:-AAPL MSFT GOOGL AMZN TSLA META NVDA}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-730}"
OUTPUT_DIR="$PROJECT_ROOT/data/processed"

mkdir -p "$OUTPUT_DIR"

export SM_OUTPUT_DIR="$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT"

echo "Fetching data for: $TICKERS"
echo "Lookback: $LOOKBACK_DAYS days"
echo "Output:   $OUTPUT_DIR"
echo ""

cd "$PROJECT_ROOT"
# shellcheck disable=SC2086
python src/ingestion/ingest_and_upload.py \
    --tickers $TICKERS \
    --lookback-days "$LOOKBACK_DAYS"

echo ""
echo "Done. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.parquet 2>/dev/null || echo "(no parquet files found)"
