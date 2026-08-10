# If we had two more weeks

## Week 1 — Production hardening

1. **Postgres + migrations (Alembic)** — partition telemetry by month; indexes on `(household_id, timestamp)`.
2. **Claim review workflow** — API for analysts to approve/reject `FLAGGED` claims, then call notarize; store reviewer, reason, and SLA.
3. **Observability** — OpenTelemetry traces on claim path; dashboard for approval rates, score distribution, and coverage warnings.
4. **Baseline lifecycle** — endpoints to request baseline reset after verified move/renovation, with cooling-off period before new claims.

## Week 2 — ML & compliance depth

1. **Shape model upgrade** — DTW distance between baseline and claim daily curves as an explicit feature; ensemble with Isolation Forest.
2. **Active learning** — queue flagged claims for labeling; periodic refit with weighted fraud examples without turning the detector fully supervised.
3. **Adversarial tests** — simulate slow drift fraud and replay attacks; regression suite on synthetic + held-out fraud patterns.
4. **Regulatory audit export** — immutable claim decision log (WORM storage or hash chain) aligned with credit registry requirements.

Together this moves the pipeline from a demonstrator to something deployable: trustworthy calculations, human-in-the-loop for edge cases, and ML that improves from operations rather than only offline synth data.
