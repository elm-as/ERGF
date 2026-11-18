"""Double Machine Learning (orthogonalized) estimator for renewable merit-order effect."""

from __future__ import annotations

from dataclasses import dataclass

from src.ingestion.grid_loader import GridObservation


@dataclass(frozen=True)
class MeritOrderEstimate:
    """Estimated causal price impact (EUR/MWh) per additional GW of renewable generation."""

    ate_eur_per_gw: float
    std_error: float
    n_observations: int


def estimate_merit_order_effect(
    observations: tuple[GridObservation, ...],
) -> MeritOrderEstimate:
    """Estimate causal merit-order effect by partialling out demand (Frisch-Waugh / DML)."""
    if len(observations) < 20:
        raise ValueError("At least 20 observations are required for DML estimation.")

    demands = [obs.demand_mw for obs in observations]
    renewables_gw = [(obs.solar_mw + obs.wind_mw) / 1000.0 for obs in observations]
    prices = [obs.spot_price_eur for obs in observations]

    res_treatment = _residualize_against_confounder(renewables_gw, demands)
    res_outcome = _residualize_against_confounder(prices, demands)

    denom = sum(t * t for t in res_treatment)
    if denom <= 1e-12:
        raise ValueError("Zero residual variation in renewable treatment after partialling out demand.")

    theta = sum(t * y for t, y in zip(res_treatment, res_outcome)) / denom
    residuals = [y - theta * t for t, y in zip(res_treatment, res_outcome)]
    sigma2 = sum(e * e for e in residuals) / max(1, len(residuals) - 2)
    std_error = (sigma2 / denom) ** 0.5
    return MeritOrderEstimate(
        ate_eur_per_gw=theta,
        std_error=std_error,
        n_observations=len(observations),
    )


def _residualize_against_confounder(target: list[float], confounder: list[float]) -> list[float]:
    """Return OLS residuals of target regressed linearly on a single confounder."""
    mean_x = sum(confounder) / len(confounder)
    mean_y = sum(target) / len(target)
    var_x = sum((x - mean_x) ** 2 for x in confounder)
    if var_x <= 1e-12:
        return [y - mean_y for y in target]

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(confounder, target)) / var_x
    intercept = mean_y - slope * mean_x
    return [y - (intercept + slope * x) for x, y in zip(confounder, target)]
