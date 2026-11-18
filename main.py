"""CLI entrypoint for the ERGF (European Renewable Grid & Price Forecasting) pipeline."""

from __future__ import annotations

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
from src.ingestion.grid_loader import SyntheticGridConfig, generate_synthetic_grid_series


def run_pipeline() -> dict[str, float]:
    """Execute end-to-end ERGF benchmark: DML causal estimate + rolling conformal forecast."""
    series = generate_synthetic_grid_series(SyntheticGridConfig(n_hours=720, seed=42))
    causal = estimate_merit_order_effect(series)
    samples = build_causal_samples(series, horizon_hours=1)
    splits = generate_rolling_splits(
        len(samples),
        RollingSplitConfig(initial_train_size=360, test_window=72, embargo_gap=24),
    )

    last_split = splits[-1]
    train_cal = samples[: last_split.train_end]
    split_point = int(len(train_cal) * 0.8)
    model = fit_price_forecaster(train_cal[:split_point])
    margin = calibrate_conformal_margin(model, train_cal[split_point:], alpha=0.10)

    test_samples = samples[last_split.test_start : last_split.test_end]
    intervals = tuple(predict_with_interval(model, s, margin) for s in test_samples)
    targets = tuple(s.target_price_eur for s in test_samples)
    points = tuple(iv.point_eur for iv in intervals)

    return {
        "merit_order_ate_eur_per_gw": round(causal.ate_eur_per_gw, 4),
        "wmape": round(compute_wmape(targets, points), 4),
        "picp_90": round(compute_interval_coverage(targets, intervals), 4),
        "pinball_50": round(compute_pinball_loss(targets, points, 0.5), 4),
    }


if __name__ == "__main__":
    metrics = run_pipeline()
    print("[ERGF] Pipeline completed:", metrics)
