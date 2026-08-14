from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.config import settings
from app.db import get_db, init_db
from app.logging_config import setup_logging
from app.ml.fraud_model import FraudScorer
from app.schemas import (
    ClaimRequest,
    ClaimResponse,
    ClaimStatus,
    HourlyBaselineProfile,
    SavingsResult,
    TelemetryBatchLoose,
    TelemetryIngestResult,
)
from app.services.baseline import compute_baseline
from app.services.claims import ingest_telemetry, process_claim
from app.services.savings import compute_savings


# ---------------------------------------------------------
# Fraud model
# ---------------------------------------------------------

fraud_scorer = FraudScorer(settings.ml_model_path)


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    init_db()

    yield


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Carbon-X Verification Pipeline",
    description=(
        "Carbon-X is an energy savings verification and "
        "fraud detection API for household energy claims."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# FRONTEND
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():
    return {
        "status": "ok",
        "service": "carbon-x",
    }


# =========================================================
# TELEMETRY
# =========================================================

@app.post(
    "/telemetry",
    response_model=TelemetryIngestResult,
    dependencies=[Depends(verify_api_key)],
    tags=["Telemetry"],
)
def post_telemetry(
    batch: TelemetryBatchLoose,
    db: Session = Depends(get_db),
):
    """
    Ingest household energy telemetry readings.
    """

    result = ingest_telemetry(
        db,
        batch.readings,
    )

    return TelemetryIngestResult(**result)


# =========================================================
# BASELINE
# =========================================================

@app.get(
    "/households/{household_id}/baseline",
    response_model=HourlyBaselineProfile,
    dependencies=[Depends(verify_api_key)],
    tags=["Verification"],
)
def get_baseline(
    household_id: str,
    db: Session = Depends(get_db),
):
    """
    Calculate the household's historical energy baseline.
    """

    try:
        hourly, kwh_day, days = compute_baseline(
            db,
            household_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    return HourlyBaselineProfile(
        household_id=household_id,
        hourly_average_watts=hourly,
        baseline_kwh_per_day=kwh_day,
        baseline_days_used=days,
    )


# =========================================================
# SAVINGS
# =========================================================

@app.get(
    "/households/{household_id}/savings",
    response_model=SavingsResult,
    dependencies=[Depends(verify_api_key)],
    tags=["Verification"],
)
def get_savings(
    household_id: str,
    period_start,
    period_end,
    db: Session = Depends(get_db),
):
    """
    Calculate verified energy and CO2e savings
    for a household over a specified period.
    """

    try:
        data = compute_savings(
            db,
            household_id,
            period_start,
            period_end,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return SavingsResult(**data)


# =========================================================
# CLAIM
# =========================================================

@app.post(
    "/households/{household_id}/claim",
    response_model=ClaimResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Claims"],
)
def post_claim(
    household_id: str,
    body: ClaimRequest,
    db: Session = Depends(get_db),
):
    """
    Process an energy-saving claim.

    The claim is evaluated using:
    - verified savings
    - fraud/anomaly score
    - fraud feature signals
    - configured approval/rejection thresholds
    - idempotency key
    """

    try:
        record, replay = process_claim(
            db,
            household_id,
            body.period_start,
            body.period_end,
            body.idempotency_key,
            fraud_scorer,
            settings.fraud_score_flag_threshold,
            settings.fraud_score_reject_threshold,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # -----------------------------------------------------
    # Fraud features
    # -----------------------------------------------------

    try:
        features = json.loads(
            record.fraud_features_json or "{}"
        )

    except (TypeError, json.JSONDecodeError):
        features = {}

    # -----------------------------------------------------
    # Claim status
    # -----------------------------------------------------

    status = ClaimStatus(record.status)

    # -----------------------------------------------------
    # Notarization eligibility
    # -----------------------------------------------------

    notarize_eligible = (
        status == ClaimStatus.AUTO_APPROVED
        or (
            status == ClaimStatus.FLAGGED_FOR_REVIEW
            and record.notarized == 1
        )
    )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return ClaimResponse(
        claim_id=record.id,
        household_id=household_id,
        status=status,
        kwh_saved=record.kwh_saved,
        co2e_saved_kg=record.co2e_saved_kg,
        legitimacy_score=record.legitimacy_score,
        fraud_features=features,
        notarize_eligible=notarize_eligible,
        idempotent_replay=replay,
    )