"""Downloader for real German/Luxembourg (DE-LU) hourly grid & spot price data via Energy-Charts API."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import requests

PRICE_URL = "https://api.energy-charts.info/price"
POWER_URL = "https://api.energy-charts.info/public_power"


def download_de_lu_hourly(
    output_csv: Path,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
) -> Path:
    """Fetch 2024 DE-LU hourly spot price, load, solar, and wind telemetry and persist as CSV."""
    if output_csv.exists():
        return output_csv

    price_df = _fetch_price_series(start_date, end_date)
    power_df = _fetch_power_series(start_date, end_date)

    merged = pd.merge(price_df, power_df, on="timestamp_utc", how="inner").dropna()
    merged = merged.sort_values("timestamp_utc").reset_index(drop=True)
    merged["hour_index"] = range(len(merged))

    cols = ["hour_index", "timestamp_utc", "demand_mw", "solar_mw", "wind_mw", "spot_price_eur"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged[cols].to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_csv


def _fetch_price_series(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly day-ahead spot prices (EUR/MWh) for bidding zone DE-LU."""
    resp = requests.get(
        PRICE_URL,
        params={"bzn": "DE-LU", "start": start_date, "end": end_date},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
        "spot_price_eur": payload["price"],
    })
    return df.set_index("timestamp_utc").resample("1h").mean().reset_index()


def _fetch_power_series(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 15-min generation & load series for Germany and resample to 1h mean MW."""
    resp = requests.get(
        POWER_URL,
        params={"country": "de", "start": start_date, "end": end_date},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    by_name = {item["name"]: item["data"] for item in payload["production_types"]}

    wind_onshore = pd.Series(by_name.get("Wind onshore", 0.0), dtype=float)
    wind_offshore = pd.Series(by_name.get("Wind offshore", 0.0), dtype=float)
    solar = pd.Series(by_name.get("Solar", 0.0), dtype=float)
    load = pd.Series(by_name.get("Load", 0.0), dtype=float)

    df = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(payload["unix_seconds"], unit="s", utc=True),
        "demand_mw": load.values,
        "solar_mw": solar.clip(lower=0.0).values,
        "wind_mw": (wind_onshore + wind_offshore).clip(lower=0.0).values,
    })
    return df.set_index("timestamp_utc").resample("1h").mean().reset_index()
