"""CLI entrypoint for ERGF using real 2024 DE-LU hourly telemetry (Energy-Charts API)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.causal.merit_order_dml import estimate_merit_order_effect
from src.evaluation.rolling_backtest import (
    RollingSplitConfig,
    compute_interval_coverage,
    compute_pinball_loss,
    compute_wmape,
    generate_rolling_splits,
)
from src.features.temporal_features import build_causal_samples
from src.forecasting.quantile_conformal import (
    calibrate_conformal_margin,
    fit_price_forecaster,
    predict_with_interval,
)
from src.ingestion.download_energy_charts import download_de_lu_hourly
from src.ingestion.grid_loader import load_grid_csv
from src.visualization.ergf_plots import generate_ergf_visuals

DATA_PATH = Path("data/de_lu_2024_hourly.csv")
RESULTS_PATH = Path("artifacts/results.json")
DASHBOARD_PATH = Path("artifacts/dashboard.html")
IMAGES_DIR = Path("images")


def run_pipeline() -> dict[str, float]:
    """Download real DE-LU 2024 data, run DML + rolling conformal forecast, and export visuals."""
    download_de_lu_hourly(DATA_PATH, start_date="2024-01-01", end_date="2024-12-31")
    series = load_grid_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    causal = estimate_merit_order_effect(series)
    samples = build_causal_samples(series, horizon_hours=1)
    splits = generate_rolling_splits(
        len(samples),
        RollingSplitConfig(initial_train_size=4320, test_window=720, embargo_gap=24),
    )

    fold_metrics: list[dict[str, float]] = []
    last_intervals = ()
    last_test_df = df.iloc[0:0]
    for idx, split in enumerate(splits, start=1):
        train_cal = samples[: split.train_end]
        cut = int(len(train_cal) * 0.8)
        model = fit_price_forecaster(train_cal[:cut])
        margin = calibrate_conformal_margin(model, train_cal[cut:], alpha=0.10)

        test_samples = samples[split.test_start : split.test_end]
        intervals = tuple(predict_with_interval(model, s, margin) for s in test_samples)
        targets = tuple(s.target_price_eur for s in test_samples)
        points = tuple(iv.point_eur for iv in intervals)

        fold_metrics.append({
            "fold": float(idx),
            "wmape": round(compute_wmape(targets, points), 4),
            "picp_90": round(compute_interval_coverage(targets, intervals), 4),
            "pinball_50": round(compute_pinball_loss(targets, points, 0.5), 4),
        })
        last_intervals = intervals
        test_indices = [s.target_index for s in test_samples]
        last_test_df = df[df["hour_index"].isin(test_indices)]

    generate_ergf_visuals(df, last_test_df, last_intervals, fold_metrics, IMAGES_DIR, DASHBOARD_PATH)
    mean_wmape = sum(m["wmape"] for m in fold_metrics) / len(fold_metrics)
    mean_picp = sum(m["picp_90"] for m in fold_metrics) / len(fold_metrics)
    mean_pinball = sum(m["pinball_50"] for m in fold_metrics) / len(fold_metrics)

    summary = {
        "n_hours": float(len(series)),
        "negative_price_hours": float((df["spot_price_eur"] < 0).sum()),
        "n_rolling_folds": float(len(splits)),
        "merit_order_ate_eur_per_gw": round(causal.ate_eur_per_gw, 4),
        "merit_order_se": round(causal.std_error, 4),
        "mean_wmape": round(mean_wmape, 4),
        "mean_picp_90": round(mean_picp, 4),
        "mean_pinball_50": round(mean_pinball, 4),
    }
    RESULTS_PATH.write_text(json.dumps({"summary": summary, "folds": fold_metrics}, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print("[ERGF] Pipeline completed:", run_pipeline())
