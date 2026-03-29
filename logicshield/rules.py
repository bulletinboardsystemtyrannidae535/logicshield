"""
LogicShield - Validation Rules
===============================
Define deterministic validation rules that LLM proposals must satisfy.

Rules are the core of the Deterministic Firewall (Layer 3 of the Truth Sandwich).
Each rule is a simple callable: given a proposal and the ground-truth state,
return True (pass) or False (fail).
"""

import re
from typing import Any, Callable, Dict, Optional


class Rule:
    """
    A single deterministic validation rule.
    
    Args:
        name: Human-readable rule name (used in error messages and ledger)
        check: callable(proposal: dict, state: dict) -> bool
        error: Error template string. Can reference {proposal} and {state}
               for dynamic error messages.
    
    Example:
        Rule("dose_safe", 
             lambda p, s: p["dose_mg"] <= s["max_dose_mg"],
             error="Dose {proposal[dose_mg]}mg exceeds maximum {state[max_dose_mg]}mg")
    """

    def __init__(self, name: str, check: Callable[[Dict, Dict], bool],
                 error: str = ""):
        self.name = name
        self._check = check
        self._error_template = error

    def evaluate(self, proposal: Dict[str, Any],
                 state: Dict[str, Any]) -> tuple:
        """
        Evaluate the rule against a proposal and state.
        
        Returns:
            (passed: bool, error_message: str)
        """
        try:
            passed = self._check(proposal, state)
        except (KeyError, TypeError, ValueError, IndexError) as e:
            return False, f"Rule '{self.name}' raised {type(e).__name__}: {e}"

        if passed:
            return True, ""

        # Build error message
        if self._error_template:
            try:
                msg = self._error_template.format(proposal=proposal, state=state)
            except (KeyError, IndexError):
                msg = self._error_template
        else:
            msg = f"Rule '{self.name}' failed."

        return False, msg

    def __repr__(self):
        return f"Rule({self.name!r})"

    # ------------------------------------------------------------------
    # Built-in rule factories
    # ------------------------------------------------------------------

    @staticmethod
    def required(key: str, error: str = "") -> "Rule":
        """Proposal must contain a non-empty value for `key`."""
        return Rule(
            name=f"required_{key}",
            check=lambda p, s: key in p and p[key] is not None and p[key] != "",
            error=error or f"'{key}' is required and must be non-empty.",
        )

    @staticmethod
    def type_check(key: str, expected_type: type, error: str = "") -> "Rule":
        """Proposal[key] must be an instance of expected_type."""
        type_name = expected_type.__name__ if isinstance(expected_type, type) else str(expected_type)
        return Rule(
            name=f"type_{key}",
            check=lambda p, s: isinstance(p.get(key), expected_type),
            error=error or f"'{key}' must be of type {type_name}.",
        )

    @staticmethod
    def range(key: str, min_val: float = None, max_val: float = None,
              error: str = "") -> "Rule":
        """Proposal[key] must be within [min_val, max_val]."""
        def check(p, s):
            val = p.get(key)
            if val is None:
                return False
            if min_val is not None and val < min_val:
                return False
            if max_val is not None and val > max_val:
                return False
            return True

        return Rule(
            name=f"range_{key}",
            check=check,
            error=error or f"'{key}' must be in range [{min_val}, {max_val}].",
        )

    @staticmethod
    def equals(key: str, state_key: str, error: str = "") -> "Rule":
        """Proposal[key] must equal state[state_key]."""
        return Rule(
            name=f"equals_{key}",
            check=lambda p, s: p.get(key) == s.get(state_key),
            error=error or f"'{key}' must equal state['{state_key}'] ({{state[{state_key}]}}).",
        )

    @staticmethod
    def less_than(key: str, state_key: str, error: str = "") -> "Rule":
        """Proposal[key] must be strictly less than state[state_key]."""
        return Rule(
            name=f"lt_{key}",
            check=lambda p, s: p.get(key) is not None and p[key] < s[state_key],
            error=error or f"'{key}' ({{proposal[{key}]}}) must be less than state['{state_key}'] ({{state[{state_key}]}}).",
        )

    @staticmethod
    def greater_than(key: str, state_key: str, error: str = "") -> "Rule":
        """Proposal[key] must be strictly greater than state[state_key]."""
        return Rule(
            name=f"gt_{key}",
            check=lambda p, s: p.get(key) is not None and p[key] > s[state_key],
            error=error or f"'{key}' ({{proposal[{key}]}}) must be greater than state['{state_key}'] ({{state[{state_key}]}}).",
        )

    @staticmethod
    def one_of(key: str, allowed: list, error: str = "") -> "Rule":
        """Proposal[key] must be one of the allowed values."""
        return Rule(
            name=f"oneof_{key}",
            check=lambda p, s: p.get(key) in allowed,
            error=error or f"'{key}' must be one of {allowed}.",
        )

    @staticmethod
    def regex(key: str, pattern: str, error: str = "") -> "Rule":
        """Proposal[key] must match the regex pattern."""
        compiled = re.compile(pattern)
        return Rule(
            name=f"regex_{key}",
            check=lambda p, s: isinstance(p.get(key), str) and compiled.match(p[key]) is not None,
            error=error or f"'{key}' must match pattern '{pattern}'.",
        )

    @staticmethod
    def custom(name: str, check: Callable[[Dict, Dict], bool],
               error: str = "") -> "Rule":
        """Create a rule with a custom check function. Alias for Rule()."""
        return Rule(name=name, check=check, error=error)
