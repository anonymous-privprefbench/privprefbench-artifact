from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resume_publication_v3_budgeted.py"
SPEC = importlib.util.spec_from_file_location("rq3_budget", SCRIPT)
rq3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = rq3
SPEC.loader.exec_module(rq3)


class PublicationV3RQ3BudgetTests(unittest.TestCase):
    def state(self, limit="1"):
        return rq3.BudgetState(
            Decimal(limit),
            Decimal("2"),
            Decimal("0.2"),
            Decimal("2.5"),
            Decimal("12"),
            10000,
        )

    def test_usage_cost_splits_cached_and_cache_write_tokens(self):
        state = self.state()
        usage = {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
        }
        self.assertEqual(state.usage_cost(usage), Decimal("0.000299"))

    def test_restore_checkpoint_counts_only_second_turn_usage(self):
        row = {
            "source_first_turn_id": "a",
            "assistant_turns": [
                {"usage": {"input_tokens": 999, "output_tokens": 999}},
                {"usage": {"input_tokens": 100, "output_tokens": 10}},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.partial"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            state = self.state()
            restored = rq3.restore_checkpoint(path, ["a", "b"], state)
        self.assertEqual(len(restored), 1)
        self.assertEqual(state.requests, 1)
        self.assertEqual(state.spent_usd, Decimal("0.00032"))

    def test_restore_checkpoint_rejects_non_prefix(self):
        row = {"source_first_turn_id": "b", "assistant_turns": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.partial"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact prefix"):
                rq3.restore_checkpoint(path, ["a", "b"], self.state())

    def test_reserve_detects_request_that_cannot_fit(self):
        class Provider:
            name = "openai"
            model = "test"
            model_version = None
            temperature = None
            reasoning_effort = "medium"
            max_output_tokens = 1000

            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools, turn_index):
                self.calls += 1

        state = self.state("0.01")
        self.assertGreater(state.reserve(Provider.max_output_tokens), state.limit_usd)

    def test_metadata_provider_never_calls_network(self):
        class First:
            provider = "openai"
            model = "test"
            model_version = None
            temperature = None
            reasoning_effort = "medium"
            max_output_tokens = 1000

        provider = rq3.MetadataProvider(First())
        self.assertEqual(provider.name, "openai")
        with self.assertRaises(AssertionError):
            provider.complete([], [], 2)


if __name__ == "__main__":
    unittest.main()
