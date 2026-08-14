from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.db import TelemetryReading
from app.services.baseline import compute_baseline, get_baseline_window_end


def _ensure_utc(dt: datetime) -> datetime:
    """
    Normalize a datetime to timezone-aware UTC.

    Database timestamps may come back as naive datetimes while
    API request timestamps may be timezone-aware. Python cannot
    compare the two directly, so everything is normalized to UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def _expected_hours(
    period_start: datetime,
    period_end: datetime,
) -> pd.DatetimeIndex:
    """
    Generate the expected hourly timestamps for the savings period.
    """
    return pd.date_range(
        start=period_start,
        end=period_end,
        freq="h",
        inclusive="left",
        tz="UTC",
    )


def compute_savings(
    db: Session,
    household_id: str,
    period_start: datetime,
    period_end: datetime,
) -> dict:
    notes: list[str] = []

    # Normalize API/request timestamps to timezone-aware UTC.
    period_start = _ensure_utc(period_start)
    period_end = _ensure_utc(period_end)

    _, baseline_kwh_per_day, _ = compute_baseline(
        db,
        household_id,
    )

    first_row = (
        db.query(TelemetryReading.timestamp)
        .filter(
            TelemetryReading.household_id == household_id
        )
        .order_by(
            TelemetryReading.timestamp.asc()
        )
        .first()
    )

    if not first_row:
        raise ValueError("No telemetry for household")

    # Normalize database timestamp to UTC.
    first_timestamp = _ensure_utc(first_row[0])

    baseline_end = get_baseline_window_end(first_timestamp)

    # get_baseline_window_end() may return a naive datetime.
    baseline_end = _ensure_utc(baseline_end)

    if period_start < baseline_end:
        notes.append(
            "Period overlaps baseline window; savings computed only on "
            "post-baseline overlap."
        )

        effective_start = max(
            period_start,
            baseline_end,
        )
    else:
        effective_start = period_start

    if effective_start >= period_end:
        raise ValueError(
            "Period does not overlap intervention window"
        )

    expected_index = _expected_hours(
        effective_start,
        period_end,
    )

    days_in_period = len(expected_index) / 24.0

    expected_kwh = (
        baseline_kwh_per_day * days_in_period
    )

    rows = (
        db.query(TelemetryReading)
        .filter(
            TelemetryReading.household_id == household_id,
            TelemetryReading.timestamp >= effective_start,
            TelemetryReading.timestamp < period_end,
        )
        .all()
    )

    if not rows:
        raise ValueError(
            "No readings in savings period"
        )

    df = pd.DataFrame(
        [
            {
                "timestamp": _ensure_utc(r.timestamp),
                "power_watts": r.power_watts,
            }
            for r in rows
        ]
    )

    # Ensure Pandas timestamps are timezone-aware UTC.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
    )

    actual_by_hour = (
        df.set_index("timestamp")["power_watts"]
    )

    aligned = actual_by_hour.reindex(
        expected_index
    )

    present_mask = aligned.notna()

    coverage_fraction = float(
        present_mask.sum() / len(expected_index)
    )

    if coverage_fraction < 0.5:
        notes.append(
            "Low data coverage in period; claim may be unreliable."
        )

    # Missing hours: impute from baseline hourly profile
    # for kWh calculation only. Missing consumption is NOT
    # treated as zero.
    hourly_baseline, _, _ = compute_baseline(
        db,
        household_id,
    )

    imputed = aligned.copy()

    for ts in expected_index[~present_mask]:
        imputed.loc[ts] = hourly_baseline[ts.hour]

    actual_kwh = float(
        imputed.sum() / 1000.0
    )

    kwh_saved = (
        expected_kwh - actual_kwh
    )

    co2e_saved_kg = (
        kwh_saved * settings.grid_emission_factor
    )

    return {
        "household_id": household_id,
        "period_start": effective_start,
        "period_end": period_end,
        "days_in_period": days_in_period,
        "expected_kwh": expected_kwh,
        "actual_kwh": actual_kwh,
        "kwh_saved": kwh_saved,
        "co2e_saved_kg": co2e_saved_kg,
        "coverage_fraction": coverage_fraction,
        "notes": notes,
    }
