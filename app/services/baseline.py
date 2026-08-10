from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.db import TelemetryReading


def _readings_to_frame(readings: list[TelemetryReading]) -> pd.DataFrame:
    if not readings:
        return pd.DataFrame(columns=["timestamp", "power_watts"])
    df = pd.DataFrame(
        [{"timestamp": r.timestamp, "power_watts": r.power_watts} for r in readings]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    return df.sort_values("timestamp")


def get_baseline_window_end(first_ts: datetime) -> datetime:
    return first_ts + timedelta(days=settings.baseline_days)


def compute_baseline(db: Session, household_id: str) -> tuple[list[float], float, int]:
    rows = (
        db.query(TelemetryReading)
        .filter(TelemetryReading.household_id == household_id)
        .order_by(TelemetryReading.timestamp.asc())
        .all()
    )
    if not rows:
        raise ValueError("No telemetry for household")

    df = _readings_to_frame(rows)
    first_ts = df["timestamp"].min()
    baseline_end = get_baseline_window_end(first_ts.to_pydatetime())
    baseline_df = df[df["timestamp"] < baseline_end]
    if baseline_df.empty:
        raise ValueError("Insufficient data for baseline period")

    baseline_df = baseline_df.copy()
    baseline_df["hour"] = baseline_df["timestamp"].dt.hour
    hourly = (
        baseline_df.groupby("hour")["power_watts"].mean().reindex(range(24), fill_value=np.nan)
    )
    if hourly.isna().any():
        hourly = hourly.fillna(hourly.mean())

    baseline_df["date"] = baseline_df["timestamp"].dt.date
    daily_kwh = baseline_df.groupby("date")["power_watts"].sum() / 1000.0
    baseline_kwh_per_day = float(daily_kwh.mean())

    days_used = len(daily_kwh)
    return [float(x) for x in hourly.tolist()], baseline_kwh_per_day, days_used


def hourly_profile_for_period(
    db: Session, household_id: str, period_start: datetime, period_end: datetime
) -> pd.DataFrame:
    rows = (
        db.query(TelemetryReading)
        .filter(
            TelemetryReading.household_id == household_id,
            TelemetryReading.timestamp >= period_start,
            TelemetryReading.timestamp < period_end,
        )
        .all()
    )
    df = _readings_to_frame(rows)
    if df.empty:
        return df
    df["hour"] = df["timestamp"].dt.hour
    return df
