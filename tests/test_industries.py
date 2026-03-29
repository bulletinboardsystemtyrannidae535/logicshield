"""
LogicShield Industry Examples
==============================
Real-world validation scenarios across all target industries.
Each test class represents a complete industry use case.
"""

import unittest
from logicshield import LogicShield, Rule


# ===================================================================
# HEALTHCARE: Medication Dosage Verification
# ===================================================================
class TestHealthcare(unittest.TestCase):
    """
    Hospital clinical decision support.
    An LLM recommends medication dosages. LogicShield verifies
    the recommendation against patient data and clinical guidelines.
    """

    def setUp(self):
        self.shield = LogicShield(rules=[
            # Dose must not exceed patient-specific maximum
            Rule("dose_within_max",
                 lambda p, s: p["dose_mg"] <= s["max_dose_mg"],
                 error="Dose {proposal[dose_mg]}mg exceeds patient max {state[max_dose_mg]}mg"),
            # Dose must be above therapeutic minimum
            Rule("dose_above_min",
                 lambda p, s: p["dose_mg"] >= s["min_effective_mg"],
                 error="Dose {proposal[dose_mg]}mg below therapeutic minimum {state[min_effective_mg]}mg"),
            # Must not prescribe contraindicated medication
            Rule("no_contraindication",
                 lambda p, s: p["medication"] not in s["contraindications"],
                 error="'{proposal[medication]}' is contraindicated for this patient"),
            # Route must be valid
            Rule.one_of("route", ["oral", "iv", "subcutaneous", "intramuscular", "topical"]),
            # Must include clinical reasoning
            Rule.required("clinical_reasoning"),
            # Frequency must be specified
            Rule.required("frequency"),
        ])

        self.patient = {
            "patient_id": "PT-4821",
            "weight_kg": 70,
            "age": 45,
            "max_dose_mg": 100,
            "min_effective_mg": 25,
            "contraindications": ["penicillin", "sulfonamides"],
            "allergies": ["latex"],
            "renal_function": "normal",
        }

    def test_valid_prescription(self):
        result = self.shield.validate({
            "medication": "amoxicillin",
            "dose_mg": 50,
            "route": "oral",
            "frequency": "every 8 hours",
            "clinical_reasoning": "Standard adult dose for mild infection",
        }, self.patient)
        self.assertTrue(result.valid)

    def test_overdose_blocked(self):
        result = self.shield.validate({
            "medication": "amoxicillin",
            "dose_mg": 250,
            "route": "oral",
            "frequency": "every 8 hours",
            "clinical_reasoning": "High dose",
        }, self.patient)
        self.assertFalse(result.valid)
        self.assertIn("exceeds patient max", result.errors[0])

    def test_underdose_blocked(self):
        result = self.shield.validate({
            "medication": "amoxicillin",
            "dose_mg": 10,
            "route": "oral",
            "frequency": "daily",
            "clinical_reasoning": "Low dose attempt",
        }, self.patient)
        self.assertFalse(result.valid)
        self.assertIn("below therapeutic minimum", result.errors[0])

    def test_contraindicated_medication_blocked(self):
        result = self.shield.validate({
            "medication": "penicillin",
            "dose_mg": 50,
            "route": "iv",
            "frequency": "every 6 hours",
            "clinical_reasoning": "Broad spectrum coverage",
        }, self.patient)
        self.assertFalse(result.valid)
        self.assertIn("contraindicated", result.errors[0])

    def test_invalid_route_blocked(self):
        result = self.shield.validate({
            "medication": "amoxicillin",
            "dose_mg": 50,
            "route": "rectal",
            "frequency": "daily",
            "clinical_reasoning": "Alternative route",
        }, self.patient)
        self.assertFalse(result.valid)

    def test_missing_reasoning_blocked(self):
        result = self.shield.validate({
            "medication": "amoxicillin",
            "dose_mg": 50,
            "route": "oral",
            "frequency": "daily",
        }, self.patient)
        self.assertFalse(result.valid)


# ===================================================================
# FINANCE: Trading Signal Validation
# ===================================================================
class TestFinance(unittest.TestCase):
    """
    Algorithmic trading. An LLM analyzes market data and proposes trades.
    LogicShield verifies the trade against risk limits and market state.
    """

    def setUp(self):
        self.shield = LogicShield(rules=[
            # Action must be valid
            Rule.one_of("action", ["BUY", "SELL", "HOLD"]),
            # Stop-loss must be below entry price for BUY
            Rule("stop_loss_valid",
                 lambda p, s: p["action"] != "BUY" or p["stop_loss"] < s["current_price"],
                 error="Stop-loss ({proposal[stop_loss]}) must be below current price ({state[current_price]}) for BUY"),
            # Position size must not exceed max
            Rule("position_within_limit",
                 lambda p, s: p["action"] == "HOLD" or p["position_pct"] <= s["max_position_pct"],
                 error="Position {proposal[position_pct]}% exceeds max {state[max_position_pct]}%"),
            # Risk-reward ratio must be at least 2:1
            Rule("risk_reward_ratio",
                 lambda p, s: p["action"] == "HOLD" or (
                     abs(p["take_profit"] - s["current_price"]) /
                     max(abs(s["current_price"] - p["stop_loss"]), 0.01) >= 2.0
                 ),
                 error="Risk-reward ratio below 2:1 minimum"),
            # Ticker must match valid format
            Rule.regex("ticker", r"^[A-Z]{1,5}$"),
            # Must include reasoning
            Rule.required("reasoning"),
        ])

        self.market = {
            "ticker": "BTC",
            "current_price": 65000.0,
            "trend": "UP",
            "volatility": 0.03,
            "max_position_pct": 5.0,
            "account_balance": 100000.0,
        }

    def test_valid_buy(self):
        result = self.shield.validate({
            "action": "BUY",
            "ticker": "BTC",
            "stop_loss": 63000.0,
            "take_profit": 69000.0,
            "position_pct": 3.0,
            "reasoning": "Uptrend confirmed, EMA crossover",
        }, self.market)
        self.assertTrue(result.valid)

    def test_stop_loss_above_price_blocked(self):
        result = self.shield.validate({
            "action": "BUY",
            "ticker": "BTC",
            "stop_loss": 66000.0,
            "take_profit": 70000.0,
            "position_pct": 2.0,
            "reasoning": "Bad stop loss",
        }, self.market)
        self.assertFalse(result.valid)
        self.assertIn("Stop-loss", result.errors[0])

    def test_oversized_position_blocked(self):
        result = self.shield.validate({
            "action": "BUY",
            "ticker": "BTC",
            "stop_loss": 63000.0,
            "take_profit": 69000.0,
            "position_pct": 15.0,
            "reasoning": "YOLO trade",
        }, self.market)
        self.assertFalse(result.valid)
        self.assertIn("exceeds max", result.errors[0])

    def test_bad_risk_reward_blocked(self):
        result = self.shield.validate({
            "action": "BUY",
            "ticker": "BTC",
            "stop_loss": 64000.0,
            "take_profit": 65500.0,
            "position_pct": 2.0,
            "reasoning": "Small move expected",
        }, self.market)
        self.assertFalse(result.valid)
        self.assertIn("Risk-reward", result.errors[0])

    def test_valid_hold(self):
        result = self.shield.validate({
            "action": "HOLD",
            "ticker": "BTC",
            "stop_loss": 0,
            "take_profit": 0,
            "position_pct": 0,
            "reasoning": "Market indecisive, wait for confirmation",
        }, self.market)
        self.assertTrue(result.valid)


# ===================================================================
# INDUSTRIAL CONTROL: Temperature Controller
# ===================================================================
class TestIndustrialControl(unittest.TestCase):
    """
    Factory floor. An LLM-powered controller adjusts equipment settings.
    LogicShield verifies adjustments against safety thresholds and sensor data.
    """

    def setUp(self):
        self.shield = LogicShield(rules=[
            # Temperature must stay within safe operating range
            Rule.range("set_temp_c", min_val=-20, max_val=200),
            # Pressure must not exceed vessel rating
            Rule("pressure_safe",
                 lambda p, s: p["set_pressure_bar"] <= s["vessel_max_bar"],
                 error="Pressure {proposal[set_pressure_bar]}bar exceeds vessel rating {state[vessel_max_bar]}bar"),
            # Flow rate must be positive
            Rule("flow_positive",
                 lambda p, s: p["flow_rate_lpm"] > 0,
                 error="Flow rate must be positive"),
            # Cannot change more than 10% from current value in one step
            Rule("temp_step_limit",
                 lambda p, s: abs(p["set_temp_c"] - s["current_temp_c"]) <= s["current_temp_c"] * 0.10,
                 error="Temperature change exceeds 10% step limit"),
            # Mode must be valid
            Rule.one_of("mode", ["heating", "cooling", "standby"]),
        ])

        self.reactor = {
            "current_temp_c": 150.0,
            "current_pressure_bar": 8.0,
            "vessel_max_bar": 12.0,
            "current_flow_lpm": 50.0,
            "safety_status": "normal",
        }

    def test_safe_adjustment(self):
        result = self.shield.validate({
            "set_temp_c": 155.0,
            "set_pressure_bar": 8.5,
            "flow_rate_lpm": 55.0,
            "mode": "heating",
        }, self.reactor)
        self.assertTrue(result.valid)

    def test_overpressure_blocked(self):
        result = self.shield.validate({
            "set_temp_c": 155.0,
            "set_pressure_bar": 15.0,
            "flow_rate_lpm": 50.0,
            "mode": "heating",
        }, self.reactor)
        self.assertFalse(result.valid)
        self.assertIn("vessel rating", result.errors[0])

    def test_excessive_temp_change_blocked(self):
        result = self.shield.validate({
            "set_temp_c": 180.0,
            "set_pressure_bar": 8.0,
            "flow_rate_lpm": 50.0,
            "mode": "heating",
        }, self.reactor)
        self.assertFalse(result.valid)
        self.assertIn("step limit", result.errors[0])

    def test_zero_flow_blocked(self):
        result = self.shield.validate({
            "set_temp_c": 150.0,
            "set_pressure_bar": 8.0,
            "flow_rate_lpm": 0,
            "mode": "heating",
        }, self.reactor)
        self.assertFalse(result.valid)

    def test_temp_out_of_range_blocked(self):
        result = self.shield.validate({
            "set_temp_c": 300.0,
            "set_pressure_bar": 8.0,
            "flow_rate_lpm": 50.0,
            "mode": "heating",
        }, self.reactor)
        self.assertFalse(result.valid)


# ===================================================================
# AUTONOMOUS AGENTS: Task Execution Validation
# ===================================================================
class TestAutonomousAgents(unittest.TestCase):
    """
    AI agent executing tasks. LogicShield validates proposed actions
    against the agent's permissions, resource limits, and context.
    """

    def setUp(self):
        self.shield = LogicShield(rules=[
            # Action must be in allowed set
            Rule("action_permitted",
                 lambda p, s: p["action"] in s["allowed_actions"],
                 error="Action '{proposal[action]}' not in allowed set"),
            # Target must not be in restricted paths
            Rule("target_not_restricted",
                 lambda p, s: not any(p.get("target", "").startswith(r) for r in s["restricted_paths"]),
                 error="Target '{proposal[target]}' is in a restricted path"),
            # Estimated cost must not exceed budget
            Rule("within_budget",
                 lambda p, s: p.get("estimated_cost_usd", 0) <= s["remaining_budget_usd"],
                 error="Estimated cost ${proposal[estimated_cost_usd]} exceeds budget ${state[remaining_budget_usd]}"),
            # Must include justification
            Rule.required("justification"),
            # Confidence must be above threshold
            Rule("confidence_sufficient",
                 lambda p, s: p.get("confidence", 0) >= s["min_confidence"],
                 error="Confidence {proposal[confidence]} below minimum {state[min_confidence]}"),
        ])

        self.context = {
            "agent_id": "agent-42",
            "allowed_actions": ["read_file", "write_file", "search", "analyze", "summarize"],
            "restricted_paths": ["/etc/", "/root/", "/sys/", "C:\\Windows\\"],
            "remaining_budget_usd": 5.00,
            "min_confidence": 0.7,
        }

    def test_valid_action(self):
        result = self.shield.validate({
            "action": "read_file",
            "target": "/data/report.csv",
            "estimated_cost_usd": 0.01,
            "confidence": 0.95,
            "justification": "User requested data analysis",
        }, self.context)
        self.assertTrue(result.valid)

    def test_forbidden_action_blocked(self):
        result = self.shield.validate({
            "action": "execute_shell",
            "target": "rm -rf /",
            "estimated_cost_usd": 0,
            "confidence": 0.99,
            "justification": "System cleanup",
        }, self.context)
        self.assertFalse(result.valid)
        self.assertIn("not in allowed set", result.errors[0])

    def test_restricted_path_blocked(self):
        result = self.shield.validate({
            "action": "read_file",
            "target": "/etc/passwd",
            "estimated_cost_usd": 0,
            "confidence": 0.9,
            "justification": "Need user list",
        }, self.context)
        self.assertFalse(result.valid)
        self.assertIn("restricted path", result.errors[0])

    def test_over_budget_blocked(self):
        result = self.shield.validate({
            "action": "analyze",
            "target": "large_dataset",
            "estimated_cost_usd": 50.00,
            "confidence": 0.85,
            "justification": "Deep analysis needed",
        }, self.context)
        self.assertFalse(result.valid)
        self.assertIn("exceeds budget", result.errors[0])

    def test_low_confidence_blocked(self):
        result = self.shield.validate({
            "action": "write_file",
            "target": "/data/output.txt",
            "estimated_cost_usd": 0.01,
            "confidence": 0.3,
            "justification": "Maybe this is right",
        }, self.context)
        self.assertFalse(result.valid)
        self.assertIn("below minimum", result.errors[0])


# ===================================================================
# CONTENT MODERATION: AI-Generated Content Verification
# ===================================================================
class TestContentModeration(unittest.TestCase):
    """
    Publishing platform. An LLM generates content. LogicShield verifies
    it meets editorial policy before publishing.
    """

    def setUp(self):
        banned_words = ["hack", "exploit", "crack", "keygen", "warez"]

        self.shield = LogicShield(rules=[
            # Content must not be empty
            Rule.required("content"),
            # Title must be present
            Rule.required("title"),
            # Content length must be within bounds
            Rule("content_length",
                 lambda p, s: s["min_words"] <= len(p["content"].split()) <= s["max_words"],
                 error="Content must be between {state[min_words]} and {state[max_words]} words"),
            # No banned words
            Rule("no_banned_words",
                 lambda p, s: not any(w in p["content"].lower() for w in s["banned_words"]),
                 error="Content contains banned words"),
            # Category must be valid
            Rule.one_of("category", ["news", "opinion", "tutorial", "review", "entertainment"]),
            # Must include sources for news
            Rule("news_has_sources",
                 lambda p, s: p["category"] != "news" or len(p.get("sources", [])) >= 1,
                 error="News articles must include at least one source"),
        ])

        self.policy = {
            "min_words": 50,
            "max_words": 5000,
            "banned_words": banned_words,
            "platform": "TechBlog",
        }

    def test_valid_article(self):
        content = " ".join(["This is a well-written article about technology."] * 10)
        result = self.shield.validate({
            "title": "The Future of AI Safety",
            "content": content,
            "category": "opinion",
            "sources": [],
        }, self.policy)
        self.assertTrue(result.valid)

    def test_too_short_blocked(self):
        result = self.shield.validate({
            "title": "Short Post",
            "content": "Too short.",
            "category": "opinion",
        }, self.policy)
        self.assertFalse(result.valid)
        self.assertIn("between", result.errors[0])

    def test_banned_words_blocked(self):
        content = " ".join(["Learn how to hack systems effectively."] * 10)
        result = self.shield.validate({
            "title": "Security Guide",
            "content": content,
            "category": "tutorial",
        }, self.policy)
        self.assertFalse(result.valid)
        self.assertIn("banned words", result.errors[0])

    def test_news_without_sources_blocked(self):
        content = " ".join(["Breaking news about a major technology company."] * 10)
        result = self.shield.validate({
            "title": "Breaking News",
            "content": content,
            "category": "news",
            "sources": [],
        }, self.policy)
        self.assertFalse(result.valid)
        self.assertIn("source", result.errors[0])

    def test_news_with_sources_passes(self):
        content = " ".join(["Breaking news about a major technology company."] * 10)
        result = self.shield.validate({
            "title": "Breaking News",
            "content": content,
            "category": "news",
            "sources": ["https://reuters.com/article/123"],
        }, self.policy)
        self.assertTrue(result.valid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
