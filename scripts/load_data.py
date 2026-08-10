"""Load parquet telemetry into SQLite via API or direct DB."""

from pathlib import Path

import pandas as pd

from app.db import SessionLocal, TelemetryReading, init_db


def load_parquet(path: Path) -> int:
    init_db()
    db = SessionLocal()
    df = pd.read_parquet(path)
    count = 0
    for row in df.itertuples(index=False):
        try:
            db.add(
                TelemetryReading(
                    household_id=row.household_id,
                    timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                    power_watts=float(row.power_watts),
                )
            )
            db.commit()
            count += 1
        except Exception:
            db.rollback()
    db.close()
    return count


if __name__ == "__main__":
    n = load_parquet(Path("data/household_telemetry.parquet"))
    print(f"Loaded {n} readings")
