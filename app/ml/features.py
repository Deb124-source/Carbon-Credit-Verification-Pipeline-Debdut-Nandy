from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sqlalchemy.orm import Session

from app.services.baseline import compute_baseline, hourly_profile_for_period


def _daily_hourly_curves(df: pd.DataFrame) -> np.ndarray:
    """Return array shape (n_days, 24) of mean watts per hour per day."""
    if df.empty:
        return np.zeros((0, 24))
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    pivot = df.pivot_table(
        index="date", columns="hour", values="power_watts", aggfunc="mean"
    )
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    pivot = pivot.reindex(columns=range(24))
    return pivot.to_numpy()


def extract_claim_features(
    db: Session, household_id: str, period_start: datetime, period_end: datetime
) -> dict[str, float]:
    hourly_baseline, baseline_kwh_day, _ = compute_baseline(db, household_id)
    baseline_vec = np.array(hourly_baseline, dtype=float)

    period_df = hourly_profile_for_period(db, household_id, period_start, period_end)
    if period_df.empty:
        return {
            "pct_reduction": 0.0,
            "shape_correlation": 0.0,
            "curve_variance_ratio": 1.0,
            "flat_zero_fraction": 1.0,
            "missing_fraction": 1.0,
            "reduction_zscore": 0.0,
        }

    period_df = period_df.copy()
    period_df["date"] = period_df["timestamp"].dt.date
    daily_kwh = period_df.groupby("date")["power_watts"].sum() / 1000.0
    actual_kwh_day = float(daily_kwh.mean()) if len(daily_kwh) else 0.0
    pct_reduction = (
        (baseline_kwh_day - actual_kwh_day) / baseline_kwh_day if baseline_kwh_day > 0 else 0.0
    )

    period_hourly = (
        period_df.groupby("hour")["power_watts"].mean().reindex(range(24)).fillna(0.0).to_numpy()
    )
    if np.std(baseline_vec) > 1e-6 and np.std(period_hourly) > 1e-6:
        shape_correlation, _ = pearsonr(baseline_vec, period_hourly)
    else:
        shape_correlation = 0.0

    from app.db import TelemetryReading
    from app.services.baseline import get_baseline_window_end

    rows = (
        db.query(TelemetryReading)
        .filter(TelemetryReading.household_id == household_id)
        .order_by(TelemetryReading.timestamp.asc())
        .all()
    )
    bdf = pd.DataFrame([{"timestamp": r.timestamp, "power_watts": r.power_watts} for r in rows])
    bdf["timestamp"] = pd.to_datetime(bdf["timestamp"])
    first_ts = bdf["timestamp"].min().to_pydatetime()
    baseline_end = get_baseline_window_end(first_ts)
    baseline_period_df = hourly_profile_for_period(db, household_id, first_ts, baseline_end)
    baseline_curves = _daily_hourly_curves(baseline_period_df)
    claim_curves = _daily_hourly_curves(period_df)
    if baseline_curves.size and claim_curves.size:
        base_var = float(np.nanvar(baseline_curves))
        claim_var = float(np.nanvar(claim_curves))
        curve_variance_ratio = claim_var / base_var if base_var > 1e-6 else 1.0
    else:
        curve_variance_ratio = 1.0

    flat_zero_fraction = float((period_df["power_watts"] <= 1.0).mean())

    expected_hours = int((period_end - period_start).total_seconds() // 3600)
    observed_hours = period_df["timestamp"].nunique()
    missing_fraction = (
        1.0 - (observed_hours / expected_hours) if expected_hours > 0 else 1.0
    )
    missing_fraction = float(np.clip(missing_fraction, 0.0, 1.0))

    # Household-relative volatility on baseline daily kWh
    bdf = baseline_period_df.copy()
    bdf["date"] = bdf["timestamp"].dt.date
    daily = bdf.groupby("date")["power_watts"].sum() / 1000.0
    vol = float(daily.std()) if len(daily) > 1 else max(baseline_kwh_day * 0.05, 1.0)
    reduction_zscore = pct_reduction * baseline_kwh_day / vol if vol > 0 else pct_reduction

    return {
        "pct_reduction": float(pct_reduction),
        "shape_correlation": float(shape_correlation),
        "curve_variance_ratio": float(curve_variance_ratio),
        "flat_zero_fraction": flat_zero_fraction,
        "missing_fraction": missing_fraction,
        "reduction_zscore": float(reduction_zscore),
    }


FEATURE_NAMES = [
    "pct_reduction",
    "shape_correlation",
    "curve_variance_ratio",
    "flat_zero_fraction",
    "missing_fraction",
    "reduction_zscore",
]
