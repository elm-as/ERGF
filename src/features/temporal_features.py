"""Causal feature engineering for multi-horizon grid forecasting."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.grid_loader import GridObservation


@dataclass(frozen=True)
class ForecastSample:
    """Strictly causal feature vector at origin t paired with target at t + horizon."""

    origin_index: int
    target_index: int
    demand_lag1_mw: float
    renewable_share: float
    residual_load_mw: float
    price_lag1_eur: float
    price_lag24_eur: float
    target_price_eur: float


def build_causal_samples(
    observations: tuple[GridObservation, ...],
    horizon_hours: int = 1,
) -> tuple[ForecastSample, ...]:
    """Construct forecasting samples using only information available at or before origin t."""
    if horizon_hours < 1:
        raise ValueError("horizon_hours must be >= 1.")
    if len(observations) <= 24 + horizon_hours:
        raise ValueError("Not enough observations to build 24h lagged causal samples.")

    samples: list[ForecastSample] = []
    for origin in range(24, len(observations) - horizon_hours):
        samples.append(_make_sample(observations, origin, horizon_hours))
    return tuple(samples)


def _make_sample(
    observations: tuple[GridObservation, ...],
    origin: int,
    horizon_hours: int,
) -> ForecastSample:
    """Assemble one ForecastSample without accessing any index > origin for features."""
    current = observations[origin]
    lag24 = observations[origin - 24]
    target = observations[origin + horizon_hours]

    renewables = current.solar_mw + current.wind_mw
    safe_demand = max(current.demand_mw, 1.0)
    return ForecastSample(
        origin_index=current.hour_index,
        target_index=target.hour_index,
        demand_lag1_mw=current.demand_mw,
        renewable_share=renewables / safe_demand,
        residual_load_mw=current.demand_mw - renewables,
        price_lag1_eur=current.spot_price_eur,
        price_lag24_eur=lag24.spot_price_eur,
        target_price_eur=target.spot_price_eur,
    )
