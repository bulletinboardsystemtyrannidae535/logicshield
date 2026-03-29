"""
LogicShield - Verification Ledger
==================================
Cryptographic audit trail for validated proposals.

Provides universal verifiability: any third party can re-run
the deterministic validation to independently prove a proposal
was safe.
"""

import hashlib
import json
from typing import Any, Dict


def compute_state_hash(state: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of the state."""
    serialized = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_signature(state_hash: str, proposal: Dict[str, Any]) -> str:
    """
    Compute verification signature: SHA-256(state_hash + proposal_json).
    
    This proves the proposal was validated against the specific state.
    """
    proposal_json = json.dumps(proposal, sort_keys=True, default=str)
    combined = f"{state_hash}:{proposal_json}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
