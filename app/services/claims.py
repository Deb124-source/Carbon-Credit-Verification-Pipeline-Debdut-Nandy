import json
import logging

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import ClaimRecord, TelemetryReading
from app.logging_config import audit_logger
from app.notarize import notarize_credit
from app.schemas import ClaimStatus, TelemetryRow
from app.services.savings import compute_savings

logger = logging.getLogger(__name__)


def ingest_telemetry(db: Session, raw_readings: list[dict]) -> dict:
    accepted = skipped_invalid = skipped_duplicate = 0
    for raw in raw_readings:
        try:
            row = TelemetryRow.model_validate(raw)
        except ValidationError as exc:
            skipped_invalid += 1
            logger.warning("Skipped invalid telemetry row: %s", exc)
            continue
        try:
            rec = TelemetryReading(
                household_id=row.household_id.strip(),
                timestamp=row.timestamp,
                power_watts=float(row.power_watts),
            )
            db.add(rec)
            db.commit()
            accepted += 1
        except IntegrityError:
            db.rollback()
            skipped_duplicate += 1
        except Exception as exc:
            db.rollback()
            skipped_invalid += 1
            logger.warning("Skipped invalid telemetry row: %s", exc)
    return {
        "accepted": accepted,
        "skipped_invalid": skipped_invalid,
        "skipped_duplicate": skipped_duplicate,
    }


def _status_from_score(score: float, flag_threshold: float, reject_threshold: float) -> ClaimStatus:
    if score >= reject_threshold:
        return ClaimStatus.REJECTED
    if score >= flag_threshold:
        return ClaimStatus.FLAGGED_FOR_REVIEW
    return ClaimStatus.AUTO_APPROVED


def process_claim(
    db: Session,
    household_id: str,
    period_start,
    period_end,
    idempotency_key: str,
    fraud_scorer,
    flag_threshold: float,
    reject_threshold: float,
) -> tuple[ClaimRecord, bool]:
    existing = (
        db.query(ClaimRecord)
        .filter(ClaimRecord.idempotency_key == idempotency_key)
        .first()
    )
    if existing:
        return existing, True

    savings = compute_savings(db, household_id, period_start, period_end)
    score, features = fraud_scorer.score_claim(
        db, household_id, savings["period_start"], savings["period_end"]
    )
    status = _status_from_score(score, flag_threshold, reject_threshold)

    record = ClaimRecord(
        household_id=household_id,
        period_start=savings["period_start"],
        period_end=savings["period_end"],
        idempotency_key=idempotency_key,
        status=status.value,
        kwh_saved=savings["kwh_saved"],
        co2e_saved_kg=savings["co2e_saved_kg"],
        legitimacy_score=score,
        fraud_features_json=json.dumps(features),
    )
    db.add(record)
    try:
        db.commit()
        db.refresh(record)
    except IntegrityError:
        db.rollback()
        existing = (
            db.query(ClaimRecord)
            .filter(ClaimRecord.idempotency_key == idempotency_key)
            .first()
        )
        if existing:
            return existing, True
        raise

    audit_logger.info(
        json.dumps(
            {
                "event": "claim_decision",
                "claim_id": record.id,
                "household_id": household_id,
                "status": record.status,
                "legitimacy_score": score,
                "kwh_saved": savings["kwh_saved"],
                "co2e_saved_kg": savings["co2e_saved_kg"],
                "features": features,
                "idempotency_key": idempotency_key,
            }
        )
    )

    if status == ClaimStatus.AUTO_APPROVED:
        tx = notarize_credit(record.id, household_id, record.co2e_saved_kg)
        record.notarized = 1
        db.commit()
        audit_logger.info(
            json.dumps(
                {
                    "event": "notarize_stub",
                    "claim_id": record.id,
                    "tx_hash": tx,
                    "eligible": True,
                }
            )
        )

    return record, False
