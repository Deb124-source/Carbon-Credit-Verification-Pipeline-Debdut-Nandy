"""
Train Isolation Forest on non-fraud synthetic households and evaluate recall/precision.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from sqlalchemy.exc import IntegrityError

from app.db import SessionLocal, TelemetryReading, init_db
from app.ml.fraud_model import FraudScorer, train_isolation_forest
from app.services.baseline import get_baseline_window_end
from app.ml.features import FEATURE_NAMES, extract_claim_features


def load_telemetry_to_db(parquet_path: Path) -> None:
    init_db()
    db = SessionLocal()
    db.query(TelemetryReading).delete()
    db.commit()
    df = pd.read_parquet(parquet_path)
    df = df.drop_duplicates(subset=["household_id", "timestamp"], keep="last")
    batch: list[TelemetryReading] = []
    chunk = 8000
    for row in df.itertuples(index=False):
        batch.append(
            TelemetryReading(
                household_id=row.household_id,
                timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
                power_watts=float(row.power_watts),
            )
        )
        if len(batch) >= chunk:
            db.bulk_save_objects(batch)
            db.commit()
            batch.clear()
    if batch:
        db.bulk_save_objects(batch)
        db.commit()
    db.close()


def build_feature_matrix(meta: pd.DataFrame, db) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_list = []
    y_list = []
    ids = []
    for row in meta.itertuples(index=False):
        first = (
            db.query(TelemetryReading.timestamp)
            .filter(TelemetryReading.household_id == row.household_id)
            .order_by(TelemetryReading.timestamp.asc())
            .first()
        )
        if not first:
            continue
        baseline_end = get_baseline_window_end(first[0])
        period_end = (
            db.query(TelemetryReading.timestamp)
            .filter(TelemetryReading.household_id == row.household_id)
            .order_by(TelemetryReading.timestamp.desc())
            .first()[0]
        )
        feats = extract_claim_features(db, row.household_id, baseline_end, period_end)
        vec = np.array([feats[n] for n in FEATURE_NAMES], dtype=float)
        vec[FEATURE_NAMES.index("shape_correlation")] = 1.0 - vec[
            FEATURE_NAMES.index("shape_correlation")
        ]
        X_list.append(vec)
        y_list.append(int(row.is_fraud_ground_truth))
        ids.append(row.household_id)
    return np.vstack(X_list), np.array(y_list), ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/household_telemetry.parquet"))
    parser.add_argument("--meta", type=Path, default=Path("data/household_metadata.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/fraud_model.joblib"))
    args = parser.parse_args()

    load_telemetry_to_db(args.data)
    meta = pd.read_csv(args.meta)

    db = SessionLocal()
    X, y, ids = build_feature_matrix(meta, db)

    train_mask = y == 0
    scaler, model = train_isolation_forest(X[train_mask])
    scorer = FraudScorer()
    scorer.scaler = scaler
    scorer.model = model
    scorer.save(str(args.model_out))

    scores = []
    for i in range(len(X)):
        x = X[i : i + 1]
        raw = -model.decision_function(scaler.transform(x))[0]
        score = float(1.0 / (1.0 + np.exp(-raw)))
        scores.append(score)

    scores = np.array(scores)
    flag_t, reject_t = 0.45, 0.75
    pred_fraud = scores >= flag_t

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred_fraud, average="binary", zero_division=0
    )
    print(f"Threshold flag={flag_t} reject={reject_t}")
    print(f"Precision (fraud as positive): {precision:.3f}")
    print(f"Recall (fraud as positive): {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"Model saved to {args.model_out}")
    db.close()


if __name__ == "__main__":
    main()
