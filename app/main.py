from contextlib import asynccontextmanager
import json

from fastapi import Depends, FastAPI, HTTPException
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

fraud_scorer = FraudScorer(settings.ml_model_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    init_db()
    yield


app = FastAPI(title="Carbon-X Verification Pipeline", lifespan=lifespan)


@app.post("/telemetry", response_model=TelemetryIngestResult, dependencies=[Depends(verify_api_key)])
def post_telemetry(batch: TelemetryBatchLoose, db: Session = Depends(get_db)):
    result = ingest_telemetry(db, batch.readings)
    return TelemetryIngestResult(**result)


@app.get(
    "/households/{household_id}/baseline",
    response_model=HourlyBaselineProfile,
    dependencies=[Depends(verify_api_key)],
)
def get_baseline(household_id: str, db: Session = Depends(get_db)):
    try:
        hourly, kwh_day, days = compute_baseline(db, household_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return HourlyBaselineProfile(
        household_id=household_id,
        hourly_average_watts=hourly,
        baseline_kwh_per_day=kwh_day,
        baseline_days_used=days,
    )


@app.get(
    "/households/{household_id}/savings",
    response_model=SavingsResult,
    dependencies=[Depends(verify_api_key)],
)
def get_savings(
    household_id: str,
    period_start,
    period_end,
    db: Session = Depends(get_db),
):
    try:
        data = compute_savings(db, household_id, period_start, period_end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SavingsResult(**data)


@app.post(
    "/households/{household_id}/claim",
    response_model=ClaimResponse,
    dependencies=[Depends(verify_api_key)],
)
def post_claim(household_id: str, body: ClaimRequest, db: Session = Depends(get_db)):
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    features = json.loads(record.fraud_features_json or "{}")
    status = ClaimStatus(record.status)
    notarize_eligible = status == ClaimStatus.AUTO_APPROVED or (
        status == ClaimStatus.FLAGGED_FOR_REVIEW and record.notarized == 1
    )
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


@app.get("/health")
def health():
    return {"status": "ok"}
