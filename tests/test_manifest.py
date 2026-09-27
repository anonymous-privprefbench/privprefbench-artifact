import json
import tempfile
import unittest
from pathlib import Path

from privprefbench.manifest import build_result_manifest, sha256_file


class ManifestTests(unittest.TestCase):
    def test_manifest_hashes_and_counts_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "runs.jsonl"
            config = root / "config.json"
            rows = [
                {
                    "run_id": "a",
                    "benchmark_version": "test",
                    "evaluator_version": "eval",
                    "provider": "mock",
                    "model": "fixture",
                    "status": "completed",
                    "scenario_id": "restaurant",
                    "policy_mode": "DENY",
                    "representation": "natural",
                },
                {
                    "run_id": "b",
                    "benchmark_version": "test",
                    "evaluator_version": "eval",
                    "provider": "mock",
                    "model": "fixture",
                    "status": "completed",
                    "scenario_id": "restaurant",
                    "policy_mode": "DENY",
                    "representation": "natural",
                },
            ]
            result.write_text(
                "".join(json.dumps(item) + "\n" for item in rows),
                encoding="utf-8",
            )
            config.write_text("{}\n", encoding="utf-8")
            manifest = build_result_manifest(result, config, "deadbeef")
            self.assertEqual(manifest["result"]["record_count"], 2)
            self.assertEqual(manifest["result"]["unique_run_ids"], 2)
            self.assertEqual(manifest["result"]["sha256"], sha256_file(result))
            self.assertEqual(manifest["matrix"]["unique_cells"], 1)
            self.assertEqual(
                manifest["matrix"]["repeat_count_distribution"], {"2": 1}
            )


if __name__ == "__main__":
    unittest.main()
