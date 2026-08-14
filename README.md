# Carbon-X Carbon Credit Verification Pipeline

Backend + ML take-home: ingest household telemetry, compute CO2e savings vs a per-household baseline, and score claims with an anomaly-detection fraud gate before notarization (stub).


## Live Link: https://carbon-credit-verification-pipeline.onrender.com/


## Quick start

### Docker

```bash
docker-compose up --build
```

API: `http://localhost:8000` — docs at `/docs`  
Header: `X-API-Key: carbon-x-dev-key`

### Local

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python generate_data.py
python scripts/train_model.py
python scripts/load_data.py
uvicorn app.main:app --reload
```

### Tests

```bash
pytest tests/ --cov=app/services --cov=app/ml --cov-report=term-missing
```

Target: **≥60% coverage** on savings + claim logic (`app/services/savings.py`, `app/services/claims.py`, `app/services/baseline.py`, `app/ml/fraud_model.py`).

## Architecture

```
telemetry CSV/parquet → SQLite
       ↓
POST /telemetry (dedupe, validate)
       ↓
GET baseline (first 30 days → 24h profile + kWh/day)
       ↓
GET savings / POST claim (CO2e vs baseline, gap-aware kWh)
       ↓
Isolation Forest on claim features → AUTO_APPROVED | FLAGGED | REJECTED
       ↓
notarize_credit() stub (AUTO_APPROVED only)
```

- **FastAPI** + **SQLAlchemy** (SQLite default)
- **ML**: unsupervised **Isolation Forest** on household-relative features (trained on non-fraud households only)
- **Audit**: structured JSON logs on every claim via `carbon_x.audit` logger

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/telemetry` | Batch ingest |
| GET | `/households/{id}/baseline` | 30-day baseline profile |
| GET | `/households/{id}/savings?period_start=&period_end=` | CO2e savings |
| POST | `/households/{id}/claim` | Savings + fraud score + decision |

Claim body: `{ "period_start", "period_end", "idempotency_key" }`.

## Savings calculation (edge cases)

- Baseline = mean daily kWh over the **first 30 days**; hourly profile for gap imputation.
- Savings period must overlap the **post-baseline** window; otherwise 400.
- **Missing hours**: imputed from the household’s baseline hourly curve (not zero). Zero-fill would under-count usage and **inflate** savings.
- `coverage_fraction` and notes flag low-confidence periods.

Formula:

`kWh_saved = baseline_kWh_per_day × days − actual_kWh`  
`CO2e_saved_kg = kWh_saved × 0.82`

## Fraud detection

### Features (household-relative)

| Feature | Why it matters |
|---------|----------------|
| `pct_reduction` | Large drops vs own baseline; context via z-score |
| `shape_correlation` | Real savings scale the daily curve; tampering often breaks morning/evening shape |
| `curve_variance_ratio` | Flat-zero fraud reduces day-to-day variance vs baseline |
| `flat_zero_fraction` | Stuck-at-zero plugs / disconnected sensors |
| `missing_fraction` | Gaming via selective reporting |
| `reduction_zscore` | Same % drop means different things for stable vs noisy homes |

### Thresholds

Legitimacy score ∈ [0, 1] from Isolation Forest decision function (sigmoid).

| Score | Status |
|-------|--------|
| &lt; 0.45 | `AUTO_APPROVED` |
| 0.45 – 0.75 | `FLAGGED_FOR_REVIEW` |
| ≥ 0.75 | `REJECTED` |

**Rationale:** contamination ~8% in training; flag band catches borderline anomalies for human review; reject catches extreme shape breaks and implausible drops seen in synthetic fraud cohort. Tuned on synthetic metadata — production would need calibration on labeled review outcomes.

Only **`AUTO_APPROVED`** triggers the notarize stub. **`FLAGGED_FOR_REVIEW`** is eligible *after* manual review (not implemented); **`REJECTED`** is not.

### Evaluation

Run `python scripts/train_model.py` after generating data. Reports precision/recall treating fraud cohort (`household_metadata.csv`) as positive — **evaluation only**, not training labels for the unsupervised model.

## Idempotency & auth

- Claims dedupe on `idempotency_key` (unique) and `(household_id, period_start, period_end)`.
- `X-API-Key` header required on all business endpoints.

## Lifestyle change false positives

Large real lifestyle shifts (move, EV, new baby) can look like fraud: same broken shape or step-change in load. In production we would:

1. Route high savings + high score to **review**, not auto-reject.
2. Collect **user-declared events** (move date, new appliance) to adjust baseline or start a new baseline window.
3. Use **semi-supervised** retraining from analyst-confirmed fraud vs approved claims.
4. Offer a **baseline reset** workflow after verified life events.

See `docs/NEXT_STEPS.md` for roadmap.

## Data

`python generate_data.py` → `data/household_telemetry.parquet`, `data/household_metadata.csv` (~200 households, 90 days hourly, 80/15/5 cohorts, 3% missing + duplicates).
