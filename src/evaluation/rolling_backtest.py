"""Rolling-origin temporal backtesting and probabilistic forecast metrics."""

from __future__ import annotations

from dataclasses import dataclass

from src.forecasting.quantile_conformal import ConformalInterval


@dataclass(frozen=True)
class RollingSplitConfig:
    """Configuration for leak-free expanding-window temporal cross-validation."""

    initial_train_size: int
    test_window: int
    embargo_gap: int = 24


@dataclass(frozen=True)
class TemporalSplitIndices:
    """Disjoint train and test index ranges separated by an embargo gap."""

    train_end: int
    test_start: int
    test_end: int


def generate_rolling_splits(
    n_samples: int,
    config: RollingSplitConfig,
) -> tuple[TemporalSplitIndices, ...]:
    """Generate expanding-window splits with strict embargo gap to prevent leakage."""
    if config.initial_train_size <= 0 or config.test_window <= 0:
        raise ValueError("Train size and test window must be strictly positive.")
    if config.embargo_gap < 0:
        raise ValueError("Embargo gap cannot be negative.")

    splits: list[TemporalSplitIndices] = []
    train_end = config.initial_train_size
    while train_end + config.embargo_gap + config.test_window <= n_samples:
        test_start = train_end + config.embargo_gap
        test_end = test_start + config.test_window
        splits.append(TemporalSplitIndices(train_end=train_end, test_start=test_start, test_end=test_end))
        train_end += config.test_window

    if not splits:
        raise ValueError("Series too short for requested rolling split configuration.")
    return tuple(splits)


def compute_pinball_loss(
    targets: tuple[float, ...],
    predictions: tuple[float, ...],
    quantile: float,
) -> float:
    """Compute mean pinball (quantile) loss at the specified quantile level."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1).")
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and equal length.")

    total = 0.0
    for y, y_hat in zip(targets, predictions):
        diff = y - y_hat
        total += max(quantile * diff, (quantile - 1.0) * diff)
    return total / len(targets)


def compute_interval_coverage(
    targets: tuple[float, ...],
    intervals: tuple[ConformalInterval, ...],
) -> float:
    """Compute empirical Prediction Interval Coverage Probability (PICP)."""
    if len(targets) != len(intervals) or not targets:
        raise ValueError("targets and intervals must be non-empty and equal length.")

    covered = sum(
        1 for y, iv in zip(targets, intervals) if iv.lower_eur <= y <= iv.upper_eur
    )
    return covered / len(targets)


def compute_wmape(targets: tuple[float, ...], predictions: tuple[float, ...]) -> float:
    """Compute Weighted Mean Absolute Percentage Error (wMAPE)."""
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and equal length.")

    denom = sum(abs(y) for y in targets)
    if denom <= 1e-12:
        raise ValueError("Sum of absolute targets is zero; wMAPE undefined.")
    num = sum(abs(y - y_hat) for y, y_hat in zip(targets, predictions))
    return num / denom
