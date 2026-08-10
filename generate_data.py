"""
Synthetic IoT telemetry for Carbon-X assignment.

Produces data/household_telemetry.parquet and data/household_metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def daily_seasonality(hours: np.ndarray, morning_peak: float, evening_peak: float) -> np.ndarray:
    """Hour-of-day multiplier: higher morning/evening, lower at night."""
    t = hours / 24.0
    base = 0.55 + 0.25 * np.sin(2 * np.pi * (t - 0.25))
    morning = morning_peak * np.exp(-0.5 * ((hours - 8) / 2.5) ** 2)
    evening = evening_peak * np.exp(-0.5 * ((hours - 19) / 3.0) ** 2)
    return base + morning + evening


def generate_household(
    household_id: str,
    cohort: str,
    rng: np.random.Generator,
    days: int = 90,
) -> pd.DataFrame:
    base_load = rng.uniform(180, 900)
    morning_peak = rng.uniform(0.15, 0.45)
    evening_peak = rng.uniform(0.2, 0.55)
    noise_scale = rng.uniform(0.04, 0.12)

    start = pd.Timestamp("2025-01-01")
    timestamps = pd.date_range(start, periods=days * 24, freq="h")
    hours = timestamps.hour.to_numpy()

    season = daily_seasonality(hours, morning_peak, evening_peak)
    watts = base_load * season
    watts *= 1.0 + rng.normal(0, noise_scale, size=len(watts))

    baseline_hours = 30 * 24
    intervention = watts.copy()

    if cohort == "genuine":
        reduction = rng.uniform(0.05, 0.20)
        intervention[baseline_hours:] *= 1.0 - reduction
    elif cohort == "no_change":
        intervention[baseline_hours:] *= 1.0 + rng.normal(0, 0.02, size=len(watts) - baseline_hours)
    elif cohort == "fraud":
        mode = rng.integers(0, 3)
        tail = intervention[baseline_hours:]
        if mode == 0:
            tail *= rng.uniform(0.05, 0.35)
        elif mode == 1:
            flat_days = rng.integers(3, 10)
            flat_start = rng.integers(0, max(1, len(tail) - flat_days * 24))
            tail[flat_start : flat_start + flat_days * 24] = rng.uniform(0, 5)
        else:
            tail *= rng.uniform(0.4, 0.7)
            fraud_hours = timestamps[baseline_hours:].hour.to_numpy()
            tail[:] = base_load * rng.uniform(0.05, 0.15) * np.ones_like(tail)
            tail += rng.normal(0, base_load * 0.02, size=len(tail))
            _ = fraud_hours
    else:
        raise ValueError(cohort)

    intervention[baseline_hours:] = np.maximum(intervention[baseline_hours:], 0)
    watts = intervention

    df = pd.DataFrame(
        {
            "household_id": household_id,
            "timestamp": timestamps,
            "power_watts": watts.astype(float),
            "cohort": cohort,
        }
    )
    return df


def inject_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    drop_idx = rng.choice(len(df), size=int(0.03 * len(df)), replace=False)
    df = df.drop(df.index[drop_idx])

    dup_idx = rng.choice(len(df), size=max(1, int(0.002 * len(df))), replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--households", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.households
    n_genuine = int(n * 0.80)
    n_no_change = int(n * 0.15)
    n_fraud = n - n_genuine - n_no_change

    cohorts = ["genuine"] * n_genuine + ["no_change"] * n_no_change + ["fraud"] * n_fraud
    rng.shuffle(cohorts)

    frames = []
    meta = []
    for i, cohort in enumerate(cohorts):
        hid = f"HH{i+1:04d}"
        frames.append(generate_household(hid, cohort, rng))
        meta.append({"household_id": hid, "cohort": cohort, "is_fraud_ground_truth": cohort == "fraud"})

    df = pd.concat(frames, ignore_index=True)
    df = inject_quality_issues(df, rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = args.out_dir / "household_telemetry.parquet"
    meta_path = args.out_dir / "household_metadata.csv"

    out_df = df[["household_id", "timestamp", "power_watts"]]
    out_df.to_parquet(telemetry_path, index=False)
    csv_path = args.out_dir / "household_telemetry.csv"
    out_df.to_csv(csv_path, index=False)
    pd.DataFrame(meta).to_csv(meta_path, index=False)

    print(f"Wrote {len(out_df)} rows -> {telemetry_path}")
    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote metadata -> {meta_path}")


if __name__ == "__main__":
    main()
