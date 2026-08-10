import pytest
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, TelemetryReading
from app.services.baseline import compute_baseline
from app.services.savings import compute_savings
from app.services.claims import ingest_telemetry, process_claim
from app.schemas import TelemetryRow, ClaimStatus
from app.ml.fraud_model import FraudScorer, _heuristic_score


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _seed_household(db, household_id: str = "HH0001", intervention_factor: float = 0.9):
    start = datetime(2025, 1, 1)
    for h in range(30 * 24):
        ts = start + timedelta(hours=h)
        hour = ts.hour
        watts = 500 + 200 * (1 if hour in (8, 19) else 0)
        db.add(
            TelemetryReading(
                household_id=household_id,
                timestamp=ts,
                power_watts=watts + (h % 5),
            )
        )
    for h in range(30 * 24, 60 * 24):
        ts = start + timedelta(hours=h)
        hour = ts.hour
        watts = (500 + 200 * (1 if hour in (8, 19) else 0)) * intervention_factor
        db.add(
            TelemetryReading(
                household_id=household_id,
                timestamp=ts,
                power_watts=watts,
            )
        )
    db.commit()


def test_baseline_profile(db):
    _seed_household(db)
    hourly, kwh_day, days = compute_baseline(db, "HH0001")
    assert len(hourly) == 24
    assert kwh_day > 0
    assert days == 30
    assert hourly[8] > hourly[3]


def test_savings_with_gap_does_not_inflate(db):
    _seed_household(db, intervention_factor=0.85)
    start = datetime(2025, 1, 31)
    end = datetime(2025, 2, 15)
    full = compute_savings(db, "HH0001", start, end)

    # Remove half the intervention readings
    db.query(TelemetryReading).filter(
        TelemetryReading.timestamp >= datetime(2025, 2, 7),
        TelemetryReading.timestamp < datetime(2025, 2, 11),
    ).delete()
    db.commit()
    gappy = compute_savings(db, "HH0001", start, end)
    assert gappy["kwh_saved"] <= full["kwh_saved"] + 1.0
    assert gappy["coverage_fraction"] < full["coverage_fraction"]


def test_telemetry_dedup(db):
    row = {
        "household_id": "HH0001",
        "timestamp": "2025-01-01T00:00:00",
        "power_watts": 100.0,
    }
    r1 = ingest_telemetry(db, [row, row])
    assert r1["accepted"] == 1
    assert r1["skipped_duplicate"] == 1


def test_claim_idempotency(db):
    _seed_household(db)
    scorer = FraudScorer(None)
    start = datetime(2025, 2, 1)
    end = datetime(2025, 2, 28)
    c1, rep1 = process_claim(db, "HH0001", start, end, "key-abc-123", scorer, 0.45, 0.75)
    c2, rep2 = process_claim(db, "HH0001", start, end, "key-abc-123", scorer, 0.45, 0.75)
    assert c1.id == c2.id
    assert rep2 is True


def test_heuristic_flags_extreme_reduction():
    score = _heuristic_score(
        {
            "pct_reduction": 0.7,
            "shape_correlation": 0.2,
            "curve_variance_ratio": 2.0,
            "flat_zero_fraction": 0.5,
            "missing_fraction": 0.1,
            "reduction_zscore": 4.0,
        }
    )
    assert score >= 0.75


def test_savings_rejects_baseline_only_period(db):
    _seed_household(db)
    with pytest.raises(ValueError):
        compute_savings(db, "HH0001", datetime(2025, 1, 1), datetime(2025, 1, 15))
