"""Ingestion and synthetic benchmark generation for European grid time series."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridObservation:
    """Immutable hourly observation of power grid demand, renewables, and spot price."""

    hour_index: int
    demand_mw: float
    solar_mw: float
    wind_mw: float
    spot_price_eur: float


@dataclass(frozen=True)
class SyntheticGridConfig:
    """Configuration for generating reproducible synthetic OPSD-like grid telemetry."""

    n_hours: int = 720
    seed: int = 42
    base_price_eur: float = 82.0


def load_grid_csv(csv_path: Path) -> tuple[GridObservation, ...]:
    """Load hourly grid observations from a CSV file with strict schema validation."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {csv_path}")

    observations: list[GridObservation] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            observations.append(_parse_grid_row(row, idx))

    if not observations:
        raise ValueError(f"Grid CSV is empty: {csv_path}")
    return tuple(observations)


def _parse_grid_row(row: dict[str, str], fallback_index: int) -> GridObservation:
    """Parse and validate a single CSV row into a GridObservation."""
    required = ("demand_mw", "solar_mw", "wind_mw", "spot_price_eur")
    missing = [key for key in required if key not in row or row[key] == ""]
    if missing:
        raise ValueError(f"Row {fallback_index} missing required fields: {missing}")

    hour_index = int(row.get("hour_index", fallback_index))
    return GridObservation(
        hour_index=hour_index,
        demand_mw=float(row["demand_mw"]),
        solar_mw=float(row["solar_mw"]),
        wind_mw=float(row["wind_mw"]),
        spot_price_eur=float(row["spot_price_eur"]),
    )


def generate_synthetic_grid_series(config: SyntheticGridConfig) -> tuple[GridObservation, ...]:
    """Generate a deterministic hourly grid series with merit-order renewable shocks."""
    if config.n_hours < 48:
        raise ValueError("n_hours must be at least 48 for lagged forecasting.")

    rng = random.Random(config.seed)
    series: list[GridObservation] = []
    for hour in range(config.n_hours):
        series.append(_simulate_hour(hour, config, rng))
    return tuple(series)


def _simulate_hour(
    hour: int,
    config: SyntheticGridConfig,
    rng: random.Random,
) -> GridObservation:
    """Simulate one hour of demand, solar, wind, and spot price."""
    diurnal = math.sin(2.0 * math.pi * (hour % 24) / 24.0)
    demand = 48_000.0 + 9_500.0 * diurnal + rng.gauss(0.0, 900.0)
    solar = max(0.0, 14_000.0 * math.sin(math.pi * max(0, (hour % 24) - 6) / 12.0))
    if (hour % 24) < 6 or (hour % 24) > 18:
        solar = 0.0
    wind = max(500.0, 11_000.0 + 4_500.0 * math.cos(2.0 * math.pi * hour / 73.0) + rng.gauss(0.0, 1_200.0))

    residual_load = demand - (solar + wind)
    # Merit-order effect: each GW of residual load raises marginal clearing price.
    spot_price = config.base_price_eur + 0.0022 * (residual_load - 30_000.0) + rng.gauss(0.0, 4.5)
    return GridObservation(
        hour_index=hour,
        demand_mw=round(demand, 2),
        solar_mw=round(solar, 2),
        wind_mw=round(wind, 2),
        spot_price_eur=round(spot_price, 2),
    )
