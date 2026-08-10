from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator


class ClaimStatus(str, Enum):
    AUTO_APPROVED = "AUTO_APPROVED"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"
    REJECTED = "REJECTED"


class TelemetryRow(BaseModel):
    household_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    power_watts: float = Field(ge=0)


class TelemetryBatchLoose(BaseModel):
    readings: list[dict[str, object]]


class TelemetryIngestResult(BaseModel):
    accepted: int
    skipped_invalid: int
    skipped_duplicate: int


class HourlyBaselineProfile(BaseModel):
    household_id: str
    hourly_average_watts: list[float]
    baseline_kwh_per_day: float
    baseline_days_used: int


class SavingsResult(BaseModel):
    household_id: str
    period_start: datetime
    period_end: datetime
    days_in_period: float
    expected_kwh: float
    actual_kwh: float
    kwh_saved: float
    co2e_saved_kg: float
    coverage_fraction: float
    notes: list[str]


class ClaimRequest(BaseModel):
    period_start: datetime
    period_end: datetime
    idempotency_key: str = Field(min_length=8, max_length=128)

    @field_validator("period_end")
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        start = info.data.get("period_start")
        if start and v <= start:
            raise ValueError("period_end must be after period_start")
        return v


class ClaimResponse(BaseModel):
    claim_id: int
    household_id: str
    status: ClaimStatus
    kwh_saved: float
    co2e_saved_kg: float
    legitimacy_score: float
    fraud_features: dict[str, Any]
    notarize_eligible: bool
    idempotent_replay: bool = False
