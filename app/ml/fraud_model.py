from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.ml.features import FEATURE_NAMES, extract_claim_features


class FraudScorer:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.scaler: StandardScaler | None = None
        self.model: IsolationForest | None = None
        if model_path and Path(model_path).exists():
            payload = joblib.load(model_path)
            self.scaler = payload["scaler"]
            self.model = payload["model"]

    def _vectorize(self, features: dict[str, float]) -> np.ndarray:
        # Higher anomaly when shape breaks, flat zeros, extreme z-scored reduction
        x = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
        # Invert correlation so low correlation = more anomalous in same direction as other flags
        x[0, FEATURE_NAMES.index("shape_correlation")] = 1.0 - x[0, FEATURE_NAMES.index("shape_correlation")]
        return x

    def score_claim(
        self, db: Session, household_id: str, period_start, period_end
    ) -> tuple[float, dict[str, float]]:
        features = extract_claim_features(db, household_id, period_start, period_end)
        if self.model is None or self.scaler is None:
            # Heuristic fallback if model not trained
            score = _heuristic_score(features)
            return score, features

        x = self._vectorize(features)
        x_scaled = self.scaler.transform(x)
        raw = -self.model.decision_function(x_scaled)[0]
        score = float(1.0 / (1.0 + np.exp(-raw)))
        return score, features

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": self.scaler, "model": self.model}, path)


def _heuristic_score(features: dict[str, float]) -> float:
    score = 0.0
    if features["pct_reduction"] > 0.5:
        score += 0.35
    if features["shape_correlation"] < 0.5:
        score += 0.25
    if features["flat_zero_fraction"] > 0.2:
        score += 0.25
    if features["reduction_zscore"] > 3.0:
        score += 0.15
    return min(score, 1.0)


def train_isolation_forest(X: np.ndarray, random_state: int = 42) -> tuple[StandardScaler, IsolationForest]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=random_state,
    )
    model.fit(Xs)
    return scaler, model
