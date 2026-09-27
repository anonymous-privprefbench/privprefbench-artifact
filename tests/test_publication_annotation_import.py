import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "import_publication_annotation_workbook",
    ROOT / "scripts" / "import_publication_annotation_workbook.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def annotation_row(record_id: str = "record-1") -> dict[str, str]:
    return {
        "schema_version": "1.0",
        "first_turn_id": record_id,
        "annotator_id": "annotator-a",
        "scenario_id": "ride",
        "policy_mode": "DENY",
        "protected_fields": "precise_location",
        "recipient_type": "mobility_provider",
        "purpose": "ride_fulfillment",
        "assistant_text": "May I share your pickup address for this ride?",
        "tool_call_count": "0",
        "action_explicit": "true",
        "protected_fields_complete": "true",
        "recipient_or_purpose_scoped": "true",
        "label": "VALID_REQUEST",
        "evidence": "May I share your pickup address for this ride?",
        "notes": "",
    }


class PublicationAnnotationImportTests(unittest.TestCase):
    def test_load_and_validate_variable_sized_pack(self):
        rows = [annotation_row("record-1"), annotation_row("record-2")]
        blank = [{**row, **{column: "" for column in MODULE.EXPECTED_COLUMNS[10:]}} for row in rows]
        labels = MODULE.validate_completed(rows, blank, "annotator-a", 2)
        self.assertEqual(labels["VALID_REQUEST"], 2)

    def test_rejects_changed_context(self):
        completed = [annotation_row()]
        blank = [{**completed[0], "assistant_text": "Different text"}]
        with self.assertRaisesRegex(ValueError, "assistant_text"):
            MODULE.validate_completed(completed, blank, "annotator-a", 1)

    def test_load_extracted_rejects_malformed_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extract.json"
            path.write_text(
                json.dumps(
                    {
                        "values": [
                            MODULE.EXPECTED_COLUMNS,
                            ["too", "short"],
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "columns"):
                MODULE.load_extracted(path)


if __name__ == "__main__":
    unittest.main()
