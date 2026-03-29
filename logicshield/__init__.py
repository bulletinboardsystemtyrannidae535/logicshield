"""
LogicShield - Deterministic Validation Firewall for LLM Outputs
================================================================

Zero dependencies. Pure Python. Works with any LLM, any domain.

    from logicshield import LogicShield, Rule

    rules = [
        Rule("dose_safe", lambda p, s: p["dose"] <= s["max_dose"],
             error="Dose exceeds maximum"),
    ]

    shield = LogicShield(rules=rules)
    result = shield.validate({"dose": 50}, {"max_dose": 100})
    assert result.valid
"""

from .shield import LogicShield, ImmutableState, FrozenNamespace
from .rules import Rule
from .result import RuleResult, ValidationResult
from .repair import repair_json
from .ledger import compute_state_hash, compute_signature

__all__ = [
    "LogicShield",
    "ImmutableState",
    "FrozenNamespace",
    "Rule",
    "RuleResult",
    "ValidationResult",
    "repair_json",
    "compute_state_hash",
    "compute_signature",
]

__version__ = "1.0.0"
