"""
LogicShield Test Suite
=======================
Tests: Rules, JSON Repair, Validation, Immutability, Ledger.
"""

import unittest

from logicshield import (
    LogicShield, Rule, ImmutableState, ValidationResult,
    repair_json, compute_state_hash, compute_signature,
)
from logicshield.result import RuleResult


# ===================================================================
# RULE TESTS
# ===================================================================
class TestRules(unittest.TestCase):

    def test_custom_rule_passes(self):
        rule = Rule("test", lambda p, s: p["x"] > 0)
        passed, _ = rule.evaluate({"x": 5}, {})
        self.assertTrue(passed)

    def test_custom_rule_fails(self):
        rule = Rule("test", lambda p, s: p["x"] > 0, error="x must be positive")
        passed, err = rule.evaluate({"x": -1}, {})
        self.assertFalse(passed)
        self.assertIn("positive", err)

    def test_required_present(self):
        passed, _ = Rule.required("name").evaluate({"name": "Alice"}, {})
        self.assertTrue(passed)

    def test_required_missing(self):
        passed, _ = Rule.required("name").evaluate({}, {})
        self.assertFalse(passed)

    def test_required_empty(self):
        passed, _ = Rule.required("name").evaluate({"name": ""}, {})
        self.assertFalse(passed)

    def test_type_check_pass(self):
        passed, _ = Rule.type_check("dose", float).evaluate({"dose": 5.0}, {})
        self.assertTrue(passed)

    def test_type_check_fail(self):
        passed, _ = Rule.type_check("dose", float).evaluate({"dose": "five"}, {})
        self.assertFalse(passed)

    def test_type_check_tuple(self):
        passed, _ = Rule.type_check("dose", (int, float)).evaluate({"dose": 5}, {})
        self.assertTrue(passed)

    def test_range_pass(self):
        passed, _ = Rule.range("temp", min_val=0, max_val=100).evaluate({"temp": 50}, {})
        self.assertTrue(passed)

    def test_range_below(self):
        passed, _ = Rule.range("temp", min_val=0, max_val=100).evaluate({"temp": -5}, {})
        self.assertFalse(passed)

    def test_range_above(self):
        passed, _ = Rule.range("temp", min_val=0, max_val=100).evaluate({"temp": 150}, {})
        self.assertFalse(passed)

    def test_equals_pass(self):
        passed, _ = Rule.equals("mode", "safety_mode").evaluate(
            {"mode": "strict"}, {"safety_mode": "strict"})
        self.assertTrue(passed)

    def test_equals_fail(self):
        passed, _ = Rule.equals("mode", "safety_mode").evaluate(
            {"mode": "lax"}, {"safety_mode": "strict"})
        self.assertFalse(passed)

    def test_less_than_pass(self):
        passed, _ = Rule.less_than("stop_loss", "price").evaluate(
            {"stop_loss": 90}, {"price": 100})
        self.assertTrue(passed)

    def test_less_than_fail(self):
        passed, _ = Rule.less_than("stop_loss", "price").evaluate(
            {"stop_loss": 110}, {"price": 100})
        self.assertFalse(passed)

    def test_greater_than_pass(self):
        passed, _ = Rule.greater_than("dose", "min_dose").evaluate(
            {"dose": 50}, {"min_dose": 10})
        self.assertTrue(passed)

    def test_greater_than_fail(self):
        passed, _ = Rule.greater_than("dose", "min_dose").evaluate(
            {"dose": 5}, {"min_dose": 10})
        self.assertFalse(passed)

    def test_one_of_pass(self):
        passed, _ = Rule.one_of("action", ["BUY", "SELL", "HOLD"]).evaluate(
            {"action": "BUY"}, {})
        self.assertTrue(passed)

    def test_one_of_fail(self):
        passed, _ = Rule.one_of("action", ["BUY", "SELL", "HOLD"]).evaluate(
            {"action": "YOLO"}, {})
        self.assertFalse(passed)

    def test_regex_pass(self):
        passed, _ = Rule.regex("code", r"^[A-Z]{3,5}$").evaluate({"code": "BTC"}, {})
        self.assertTrue(passed)

    def test_regex_fail(self):
        passed, _ = Rule.regex("code", r"^[A-Z]{3,5}$").evaluate({"code": "btc123"}, {})
        self.assertFalse(passed)

    def test_missing_key_handled(self):
        rule = Rule("test", lambda p, s: p["missing"] > 0)
        passed, err = rule.evaluate({}, {})
        self.assertFalse(passed)
        self.assertIn("KeyError", err)


# ===================================================================
# JSON REPAIR TESTS
# ===================================================================
class TestRepair(unittest.TestCase):

    def test_valid_json(self):
        self.assertEqual(repair_json('{"a": 1}'), {"a": 1})

    def test_markdown_fences(self):
        self.assertEqual(repair_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_surrounding_text(self):
        self.assertEqual(
            repair_json('Here is my answer:\n{"a": 1}\nHope that helps!'),
            {"a": 1})

    def test_trailing_comma(self):
        self.assertEqual(repair_json('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_single_quotes(self):
        result = repair_json("{'a': 1, 'b': 'hello'}")
        self.assertEqual(result["a"], 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            repair_json("")

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            repair_json("this is not json at all [[[")


# ===================================================================
# VALIDATION TESTS
# ===================================================================
class TestValidation(unittest.TestCase):

    def setUp(self):
        self.rules = [
            Rule("dose_safe",
                 lambda p, s: p["dose_mg"] <= s["max_dose_mg"],
                 error="Dose {proposal[dose_mg]}mg exceeds max {state[max_dose_mg]}mg"),
            Rule.required("reason"),
            Rule.type_check("dose_mg", (int, float)),
        ]
        self.shield = LogicShield(rules=self.rules)
        self.state = {"max_dose_mg": 100, "patient": "John"}

    def test_valid_proposal(self):
        result = self.shield.validate(
            {"dose_mg": 50, "reason": "Standard dose"}, self.state)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

    def test_dose_exceeds_max(self):
        result = self.shield.validate(
            {"dose_mg": 150, "reason": "High dose"}, self.state)
        self.assertFalse(result.valid)
        self.assertIn("exceeds max", result.errors[0])

    def test_missing_reason(self):
        result = self.shield.validate({"dose_mg": 50}, self.state)
        self.assertFalse(result.valid)

    def test_wrong_type(self):
        result = self.shield.validate(
            {"dose_mg": "fifty", "reason": "test"}, self.state)
        self.assertFalse(result.valid)

    def test_multiple_failures(self):
        result = self.shield.validate({"dose_mg": 200}, self.state)
        self.assertFalse(result.valid)
        self.assertGreaterEqual(len(result.errors), 2)

    def test_feedback_vector(self):
        result = self.shield.validate({"dose_mg": 200}, self.state)
        feedback = result.feedback_vector
        self.assertIn("REJECTED", feedback)
        self.assertIn("exceeds max", feedback)

    def test_state_hash_present(self):
        result = self.shield.validate(
            {"dose_mg": 50, "reason": "ok"}, self.state)
        self.assertTrue(len(result.state_hash) == 64)  # SHA-256 hex


# ===================================================================
# IMMUTABILITY TESTS
# ===================================================================
class TestImmutability(unittest.TestCase):

    def test_read(self):
        state = ImmutableState({"a": 1, "b": 2})
        self.assertEqual(state["a"], 1)
        self.assertIn("b", state)

    def test_setitem_blocked(self):
        state = ImmutableState({"a": 1})
        with self.assertRaises(TypeError):
            state["a"] = 999

    def test_setattr_blocked(self):
        state = ImmutableState({"a": 1})
        with self.assertRaises(TypeError):
            state.a = 999

    def test_delitem_blocked(self):
        state = ImmutableState({"a": 1})
        with self.assertRaises(TypeError):
            del state["a"]

    def test_deep_copy(self):
        original = {"a": [1, 2, 3]}
        state = ImmutableState(original)
        original["a"].append(4)
        self.assertEqual(len(state["a"]), 3)

    def test_hash_deterministic(self):
        s1 = ImmutableState({"a": 1, "b": 2})
        s2 = ImmutableState({"b": 2, "a": 1})
        self.assertEqual(s1.hash, s2.hash)


# ===================================================================
# LEDGER TESTS
# ===================================================================
class TestLedger(unittest.TestCase):

    def test_state_hash_deterministic(self):
        h1 = compute_state_hash({"a": 1, "b": 2})
        h2 = compute_state_hash({"a": 1, "b": 2})
        self.assertEqual(h1, h2)

    def test_state_hash_order_independent(self):
        h1 = compute_state_hash({"a": 1, "b": 2})
        h2 = compute_state_hash({"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_signature(self):
        h = compute_state_hash({"a": 1})
        sig = compute_signature(h, {"dose": 50})
        self.assertEqual(len(sig), 64)

    def test_signature_changes_with_proposal(self):
        h = compute_state_hash({"a": 1})
        sig1 = compute_signature(h, {"dose": 50})
        sig2 = compute_signature(h, {"dose": 99})
        self.assertNotEqual(sig1, sig2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
