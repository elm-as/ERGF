"""Publication-grade 300 DPI matplotlib charts and interactive Plotly dashboard for ERGF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.forecasting.quantile_conformal import ConformalInterval


def generate_ergf_visuals(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    intervals: tuple[ConformalInterval, ...],
    fold_metrics: list[dict[str, float]],
    images_dir: Path,
    dashboard_path: Path,
) -> None:
    """Render 3 PNG figures (300 DPI) and a standalone Plotly HTML5 dashboard."""
    images_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

    _plot_merit_order_curve(df, images_dir / "ergf_merit_order_causal.png")
    _plot_conformal_forecast(test_df, intervals, images_dir / "ergf_conformal_forecast.png")
    _plot_rolling_folds(fold_metrics, images_dir / "ergf_rolling_backtest_metrics.png")
    _build_plotly_dashboard(df, test_df, intervals, dashboard_path)


def _plot_merit_order_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ren_gw = (df["solar_mw"] + df["wind_mw"]) / 1000.0
    sample = df.sample(min(2500, len(df)), random_state=42)
    ren_s = (sample["solar_mw"] + sample["wind_mw"]) / 1000.0

    sc = ax.scatter(ren_s, sample["spot_price_eur"], c=sample["demand_mw"] / 1000.0,
                    cmap="viridis", alpha=0.45, s=14, edgecolors="none")
    z = np.polyfit(ren_gw, df["spot_price_eur"], 1)
    xs = np.linspace(ren_gw.min(), ren_gw.max(), 100)
    ax.plot(xs, np.polyval(z, xs), color="#DC2626", linewidth=2.2, label=f"Merit-Order Slope: {z[0]:.2f} EUR/MWh/GW")
    ax.axhline(0.0, color="#64748B", linestyle="--", linewidth=1.0, label="Negative Price Threshold (0 EUR)")
    plt.colorbar(sc, ax=ax, label="Grid Demand (GW)")
    ax.set_title("DE-LU 2024: Renewable Generation vs Day-Ahead Spot Price", fontweight="bold")
    ax.set_xlabel("Total Renewable Generation (Solar + Wind, GW)")
    ax.set_ylabel("Spot Price (EUR/MWh)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_conformal_forecast(
    test_df: pd.DataFrame,
    intervals: tuple[ConformalInterval, ...],
    out_path: Path,
) -> None:
    n_show = min(168, len(test_df))
    sub = test_df.iloc[-n_show:]
    ivs = intervals[-n_show:]
    x = np.arange(n_show)
    lower = [iv.lower_eur for iv in ivs]
    upper = [iv.upper_eur for iv in ivs]
    point = [iv.point_eur for iv in ivs]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.fill_between(x, lower, upper, color="#93C5FD", alpha=0.4, label="90% Conformal Interval")
    ax.plot(x, sub["spot_price_eur"].values, color="#0F172A", linewidth=1.6, label="Actual DE-LU Price")
    ax.plot(x, point, color="#2563EB", linewidth=1.3, linestyle="--", label="Point Forecast")
    ax.set_title("7-Day Out-of-Sample Spot Price Forecast with Conformal Bounds", fontweight="bold")
    ax.set_xlabel("Hour (Final Test Fold)")
    ax.set_ylabel("Spot Price (EUR/MWh)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_rolling_folds(fold_metrics: list[dict[str, float]], out_path: Path) -> None:
    folds = [f"Fold {int(m['fold'])}" for m in fold_metrics]
    picps = [m["picp_90"] * 100.0 for m in fold_metrics]
    pinballs = [m["pinball_50"] for m in fold_metrics]

    fig, ax1 = plt.subplots(figsize=(8.0, 4.6))
    ax1.bar(folds, picps, color="#10B981", alpha=0.8, width=0.45, label="PICP 90% (%)")
    ax1.axhline(90.0, color="#047857", linestyle="--", linewidth=1.2, label="Target Coverage (90%)")
    ax1.set_ylabel("Empirical Coverage (%)", color="#047857")
    ax1.set_ylim(0, 108)

    ax2 = ax1.twinx()
    ax2.plot(folds, pinballs, color="#F59E0B", marker="o", linewidth=2.0, label="Pinball Loss (q50)")
    ax2.set_ylabel("Pinball Loss (EUR/MWh)", color="#B45309")
    ax1.set_title("Rolling-Origin Cross-Validation Stability (24h Embargo)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_plotly_dashboard(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    intervals: tuple[ConformalInterval, ...],
    dashboard_path: Path,
) -> None:
    fig = make_subplots(rows=2, cols=1, subplot_titles=("DE-LU 2024 Daily Mean Spot Price vs Renewables", "Final Fold Conformal Forecast (168h)"))
    daily = df.copy()
    daily["date"] = pd.to_datetime(daily["timestamp_utc"]).dt.date
    grp = daily.groupby("date", as_index=False).mean(numeric_only=True)

    fig.add_trace(go.Scatter(x=grp["date"], y=grp["spot_price_eur"], name="Spot Price (EUR/MWh)", line=dict(color="#1D4ED8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=grp["date"], y=(grp["solar_mw"] + grp["wind_mw"]) / 1000.0, name="Renewables (GW)", line=dict(color="#10B981")), row=1, col=1)

    n_show = min(168, len(test_df))
    sub = test_df.iloc[-n_show:]
    ivs = intervals[-n_show:]
    fig.add_trace(go.Scatter(x=sub["timestamp_utc"], y=sub["spot_price_eur"], name="Actual Price", line=dict(color="#0F172A")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sub["timestamp_utc"], y=[iv.point_eur for iv in ivs], name="Forecast", line=dict(color="#2563EB", dash="dash")), row=2, col=1)
    fig.update_layout(height=760, title_text="ERGF — European Renewable Grid & Price Forecasting (DE-LU 2024)", template="plotly_white")
    fig.write_html(str(dashboard_path), include_plotlyjs="cdn")
