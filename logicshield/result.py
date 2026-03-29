"""
LogicShield - Result Types
==========================
Data classes for validation results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RuleResult:
    """Result of a single rule check."""
    name: str
    passed: bool
    error: str = ""


@dataclass
class ValidationResult:
    """
    Result of validating a proposal against all rules.
    
    Attributes:
        valid: Whether all rules passed
        proposal: The original proposal that was validated
        rule_results: Individual result for each rule
        errors: List of error messages from failed rules
        state_hash: SHA-256 hash of the frozen input state
    """
    valid: bool
    proposal: Dict[str, Any]
    rule_results: List[RuleResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    state_hash: str = ""

    @property
    def failed_rules(self) -> List[RuleResult]:
        return [r for r in self.rule_results if not r.passed]

    @property
    def feedback_vector(self) -> str:
        """
        Build the Feedback Error Vector.
        
        Use this to feed specific errors back into your LLM prompt
        for correction. This is the bridge between LogicShield and
        your own retry loop.
        """
        if not self.errors:
            return ""
        lines = ["[SYSTEM FEEDBACK] Your proposal was REJECTED. Fix these errors:"]
        for i, err in enumerate(self.errors, 1):
            lines.append(f"  {i}. {err}")
        return "\n".join(lines)
