"""Stub for blockchain notarization gate."""

import logging

logger = logging.getLogger(__name__)


def notarize_credit(claim_id: int, household_id: str, co2e_saved_kg: float) -> str:
    tx_hash = f"0xstub-{household_id}-{claim_id}-{int(co2e_saved_kg * 100)}"
    logger.info(
        "Notarize stub invoked claim_id=%s household=%s co2e_kg=%.2f tx=%s",
        claim_id,
        household_id,
        co2e_saved_kg,
        tx_hash,
    )
    return tx_hash
