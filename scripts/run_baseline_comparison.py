"""Run baseline model comparison against the trained XGBoost model.

Loads all processed parquet files, evaluates three baseline models using the
same walk-forward CV and test split as the XGBoost training pipeline, then
prints a side-by-side comparison table.

Usage:
    python scripts/run_baseline_comparison.py
    python scripts/run_baseline_comparison.py --data-dir data/processed
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.baselines import (
    NaiveMeanBaseline,
    LinearRegressionBaseline,
    RidgeRegressionBaseline,
    evaluate_baselines,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

XGBOOST_METRICS_PATH = Path("output/metrics.json")


def load_data(data_dir: Path) -> pd.DataFrame:
    parquets = sorted(data_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames = []
    for p in parquets:
        df = pd.read_parquet(p)
        if "target" not in df.columns:
            continue
        frames.append(df)

    combined = pd.concat(frames).sort_index()
    print(f"Loaded {len(combined):,} rows from {len(frames)} file(s) in {data_dir}")
    return combined


def load_xgboost_metrics() -> dict | None:
    if not XGBOOST_METRICS_PATH.exists():
        return None
    with open(XGBOOST_METRICS_PATH) as f:
        raw = json.load(f)
    # train_xgboost.py writes {"cv_metrics": {...}, "test_metrics": {...}, ...}
    return {
        "cv_avg": raw.get("cv_metrics", {}),
        "test": raw.get("test_metrics", {}),
    }


def print_table(results: list[dict], xgb: dict | None) -> None:
    try:
        from tabulate import tabulate
    except ImportError:
        print("\nInstall tabulate for formatted output: pip install tabulate\n")
        tabulate = None

    cols = ["model", "split", "rmse", "mae", "directional_accuracy", "sharpe_proxy", "n_samples"]
    test_rows = [r for r in results if r["split"] == "test"]

    # Append XGBoost rows from saved metrics.json
    if xgb:
        test_rows.append({
            "model": "xgboost",
            "split": "test",
            **xgb["test"],
        })

    rows = [[r.get(c, "") for c in cols] for r in test_rows]
    headers = ["Model", "Split", "RMSE", "MAE", "Dir. Acc.", "Sharpe", "N"]

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON — TEST SET HOLDOUT")
    print("=" * 70)
    if tabulate:
        print(tabulate(rows, headers=headers, floatfmt=".4f", tablefmt="github"))
    else:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(v) for v in row))

    print()
    print("Notes:")
    print("  - Dir. Acc.: fraction of correct up/down direction predictions")
    print("  - Sharpe: annualised Sharpe ratio using model signal as position")
    print("  - XGBoost metrics loaded from output/metrics.json (saved during training)")
    print("  - All models use the same 90/10 temporal train/test split")

    # Also print CV averages
    cv_rows = [r for r in results if r["split"] == "cv_avg"]
    if xgb:
        cv_rows.append({"model": "xgboost", "split": "cv_avg", **xgb["cv_avg"]})

    cv_table = [[r.get(c, "") for c in cols] for r in cv_rows]
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON — CROSS-VALIDATION AVERAGE (3-fold walk-forward)")
    print("=" * 70)
    if tabulate:
        print(tabulate(cv_table, headers=headers, floatfmt=".4f", tablefmt="github"))
    else:
        for row in cv_table:
            print("\t".join(str(v) for v in row))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline models vs XGBoost")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with processed parquet files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    df = load_data(data_dir)

    baselines = [
        NaiveMeanBaseline(),
        LinearRegressionBaseline(),
        RidgeRegressionBaseline(),
    ]

    print("Running walk-forward CV + test evaluation for 3 baseline models...")
    results = evaluate_baselines(df, baselines)

    xgb_metrics = load_xgboost_metrics()
    if xgb_metrics is None:
        print("Warning: output/metrics.json not found — XGBoost row will be omitted from table")

    print_table(results, xgb_metrics)


if __name__ == "__main__":
    main()
