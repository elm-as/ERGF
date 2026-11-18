"""Quantile baseline forecasting with split conformal prediction intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.features.temporal_features import ForecastSample


@dataclass(frozen=True)
class LinearPriceModel:
    """Immutable ridge-regularized linear forecaster for spot prices."""

    intercept: float
    coef_residual_load: float
    coef_price_lag1: float
    coef_price_lag24: float


@dataclass(frozen=True)
class ConformalInterval:
    """Calibrated prediction with symmetric conformal bounds."""

    point_eur: float
    lower_eur: float
    upper_eur: float


def fit_price_forecaster(samples: tuple[ForecastSample, ...]) -> LinearPriceModel:
    """Fit a deterministic least-squares forecaster on training ForecastSamples."""
    if len(samples) < 10:
        raise ValueError("At least 10 samples are required to fit the forecaster.")

    mean_y = sum(s.target_price_eur for s in samples) / len(samples)
    mean_res = sum(s.residual_load_mw for s in samples) / len(samples)
    mean_lag1 = sum(s.price_lag1_eur for s in samples) / len(samples)
    mean_lag24 = sum(s.price_lag24_eur for s in samples) / len(samples)

    # Normal-equation-free coordinate descent keeps the core dependency-free and bounded.
    weights = [0.0, 0.0, 0.0]
    for _ in range(40):
        weights = _update_weights(samples, weights, (mean_y, mean_res, mean_lag1, mean_lag24))

    intercept = mean_y - (
        weights[0] * mean_res + weights[1] * mean_lag1 + weights[2] * mean_lag24
    )
    return LinearPriceModel(
        intercept=intercept,
        coef_residual_load=weights[0],
        coef_price_lag1=weights[1],
        coef_price_lag24=weights[2],
    )


def _update_weights(
    samples: tuple[ForecastSample, ...],
    weights: list[float],
    means: tuple[float, float, float, float],
) -> list[float]:
    """Perform one regularized gradient step on centered features."""
    mean_y, mean_res, mean_lag1, mean_lag24 = means
    grad = [0.0, 0.0, 0.0]
    scale_res = 10_000.0
    for s in samples:
        x0 = (s.residual_load_mw - mean_res) / scale_res
        x1 = (s.price_lag1_eur - mean_lag1) / 50.0
        x2 = (s.price_lag24_eur - mean_lag24) / 50.0
        err = (weights[0] * scale_res * x0 + weights[1] * 50.0 * x1 + weights[2] * 50.0 * x2) - (
            s.target_price_eur - mean_y
        )
        grad[0] += err * x0 / len(samples)
        grad[1] += err * x1 / len(samples)
        grad[2] += err * x2 / len(samples)

    return [
        weights[0] - 0.0001 * grad[0],
        weights[1] - 0.005 * grad[1],
        weights[2] - 0.005 * grad[2],
    ]


def predict_point(model: LinearPriceModel, sample: ForecastSample) -> float:
    """Predict spot price for a single ForecastSample."""
    return (
        model.intercept
        + model.coef_residual_load * sample.residual_load_mw
        + model.coef_price_lag1 * sample.price_lag1_eur
        + model.coef_price_lag24 * sample.price_lag24_eur
    )


def calibrate_conformal_margin(
    model: LinearPriceModel,
    calibration_samples: tuple[ForecastSample, ...],
    alpha: float = 0.10,
) -> float:
    """Compute the finite-sample split-conformal quantile margin at level 1 - alpha."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if not calibration_samples:
        raise ValueError("Calibration set cannot be empty.")

    residuals = sorted(
        abs(s.target_price_eur - predict_point(model, s)) for s in calibration_samples
    )
    n = len(residuals)
    rank = min(n - 1, math.ceil((n + 1) * (1.0 - alpha)) - 1)
    return residuals[max(0, rank)]


def predict_with_interval(
    model: LinearPriceModel,
    sample: ForecastSample,
    margin: float,
) -> ConformalInterval:
    """Return a point forecast wrapped in calibrated symmetric conformal bounds."""
    if margin < 0.0:
        raise ValueError("Conformal margin cannot be negative.")
    point = predict_point(model, sample)
    return ConformalInterval(
        point_eur=point,
        lower_eur=point - margin,
        upper_eur=point + margin,
    )
