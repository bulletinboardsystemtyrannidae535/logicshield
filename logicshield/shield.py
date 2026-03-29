"""
LogicShield - The Truth Sandwich
=================================
Deterministic validation firewall for LLM outputs.

Validates AI-generated proposals against ground-truth state using
deterministic rules. Ensures no LLM output reaches production without
passing mathematically verifiable checks.

Zero dependencies. Pure Python. Works with any LLM, any domain.

Copyright (c) 2026 Mattijs Moens. All rights reserved.
Licensed under the Business Source License 1.1.
"""

import copy
import logging
from typing import Any, Dict, List

from .rules import Rule
from .result import RuleResult, ValidationResult
from .ledger import compute_state_hash

logger = logging.getLogger(__name__)


# ===================================================================
# FROZEN NAMESPACE (Immutability Enforcement)
# ===================================================================
class FrozenNamespace(type):
    """
    Metaclass that prevents modification of class attributes.
    Makes safety-critical constants truly immutable in memory.
    """
    def __setattr__(cls, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
            return
        raise TypeError(
            f"LOGICSHIELD VIOLATION: Cannot modify frozen attribute '{key}'"
        )

    def __delattr__(cls, key):
        raise TypeError(
            f"LOGICSHIELD VIOLATION: Cannot delete frozen attribute '{key}'"
        )


# ===================================================================
# IMMUTABLE STATE WRAPPER
# ===================================================================
class ImmutableState:
    """
    Frozen wrapper around the input state dict.
    
    Deep-copies the state at creation time and prevents all modification.
    This is the Input Anchor -- the ground truth is locked to this
    snapshot and cannot be tampered with during validation.
    """

    def __init__(self, state: Dict[str, Any]):
        object.__setattr__(self, "_data", copy.deepcopy(state))
        object.__setattr__(self, "_hash", compute_state_hash(state))

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __setattr__(self, key, value):
        raise TypeError("ImmutableState cannot be modified.")

    def __setitem__(self, key, value):
        raise TypeError("ImmutableState cannot be modified.")

    def __delitem__(self, key):
        raise TypeError("ImmutableState cannot be modified.")

    def to_dict(self) -> dict:
        """Return a deep copy of the state data."""
        return copy.deepcopy(self._data)

    @property
    def hash(self) -> str:
        return self._hash

    def __repr__(self):
        return f"ImmutableState({self._data})"


# ===================================================================
# THE LOGIC SHIELD
# ===================================================================
class LogicShield(metaclass=FrozenNamespace):
    """
    Deterministic validation firewall for LLM outputs.
    
    Takes a proposal (from any source) and validates it against
    ground-truth state using deterministic rules. Returns a clear
    pass/fail verdict with detailed error messages.
    
    Usage:
        shield = LogicShield(rules=[
            Rule("dose_safe", lambda p, s: p["dose_mg"] <= s["max_dose_mg"],
                 error="Dose exceeds maximum"),
            Rule.required("reason"),
        ])
        
        result = shield.validate(
            proposal={"dose_mg": 50, "reason": "Standard dose"},
            state={"max_dose_mg": 100},
        )
        
        result.valid    # True
        result.errors   # []
    """

    def __init__(self, rules: List[Rule]):
        """
        Args:
            rules: List of Rule objects defining the Deterministic Firewall.
        """
        self._rules = list(rules)

    def validate(self, proposal: Dict[str, Any],
                 state: Dict[str, Any]) -> ValidationResult:
        """
        Validate a proposal against all rules using the ground-truth state.
        
        This is the Deterministic Firewall. Given a proposal (from any
        LLM, user input, API, etc.) and the ground-truth state, every
        rule is evaluated. The proposal passes only if ALL rules pass.
        
        Args:
            proposal: The AI-generated proposal (dict)
            state: The ground-truth state (dict)
            
        Returns:
            ValidationResult with .valid, .errors, .rule_results,
            .feedback_vector, .state_hash, .signature
        """
        frozen = ImmutableState(state)
        state_dict = frozen.to_dict()
        rule_results = []
        errors = []

        for rule in self._rules:
            passed, error = rule.evaluate(proposal, state_dict)
            rule_results.append(RuleResult(
                name=rule.name,
                passed=passed,
                error=error,
            ))
            if not passed:
                errors.append(error)

        return ValidationResult(
            valid=len(errors) == 0,
            proposal=proposal,
            rule_results=rule_results,
            errors=errors,
            state_hash=frozen.hash,
        )

    @property
    def rules(self) -> List[Rule]:
        """Return the list of rules (read-only copy)."""
        return list(self._rules)

    def __repr__(self):
        return f"LogicShield(rules={len(self._rules)})"
