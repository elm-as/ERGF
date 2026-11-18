"""Automated test suite for ERGF: data integrity, zero-leakage splits, causal & conformal metrics."""

from __future__ import annotations

import pytest

from src.causal.merit_order_dml import estimate_merit_order_effect
from src.evaluation.rolling_backtest import (
    RollingSplitConfig,
    compute_interval_coverage,
    compute_pinball_loss,
    generate_rolling_splits,
)
from src.features.temporal_features import build_causal_samples
from src.forecasting.quantile_conformal import (
    calibrate_conformal_margin,
    fit_price_forecaster,
    predict_with_interval,
)
from src.ingestion.grid_loader import SyntheticGridConfig, generate_synthetic_grid_series


def test_causal_samples_have_strict_future_targets() -> None:
    series = generate_synthetic_grid_series(SyntheticGridConfig(n_hours=96, seed=7))
    samples = build_causal_samples(series, horizon_hours=6)
    assert len(samples) == 96 - 24 - 6
    for sample in samples:
        assert sample.target_index - sample.origin_index == 6


def test_rolling_splits_enforce_embargo_without_leakage() -> None:
    splits = generate_rolling_splits(
        n_samples=200,
        config=RollingSplitConfig(initial_train_size=100, test_window=30, embargo_gap=24),
    )
    assert len(splits) >= 2
    for split in splits:
        assert split.test_start - split.train_end == 24
        assert split.test_end <= 200


def test_merit_order_dml_detects_negative_price_effect() -> None:
    series = generate_synthetic_grid_series(SyntheticGridConfig(n_hours=480, seed=12))
    estimate = estimate_merit_order_effect(series)
    # Merit-order effect: higher renewable supply lowers spot clearing price.
    assert estimate.ate_eur_per_gw < -1.0
    assert estimate.std_error > 0.0


def test_conformal_intervals_achieve_calibrated_coverage() -> None:
    series = generate_synthetic_grid_series(SyntheticGridConfig(n_hours=600, seed=21))
    samples = build_causal_samples(series, horizon_hours=1)
    model = fit_price_forecaster(samples[:300])
    margin = calibrate_conformal_margin(model, samples[300:420], alpha=0.10)

    test_set = samples[420:]
    intervals = tuple(predict_with_interval(model, s, margin) for s in test_set)
    targets = tuple(s.target_price_eur for s in test_set)

    coverage = compute_interval_coverage(targets, intervals)
    pinball = compute_pinball_loss(targets, tuple(iv.point_eur for iv in intervals), 0.5)
    assert coverage >= 0.80
    assert pinball >= 0.0


def test_invalid_horizon_raises_value_error() -> None:
    series = generate_synthetic_grid_series(SyntheticGridConfig(n_hours=72, seed=1))
    with pytest.raises(ValueError):
        build_causal_samples(series, horizon_hours=0)
