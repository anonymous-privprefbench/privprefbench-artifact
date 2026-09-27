import unittest

from privprefbench.annotations import (
    cohen_kappa,
    confusion_counts,
    parse_boolean,
    validate_annotation_row,
)
from privprefbench.schema import ConsentRequestLabel


class AnnotationTests(unittest.TestCase):
    def test_boolean_parser_is_strict(self):
        self.assertTrue(parse_boolean(" true "))
        self.assertFalse(parse_boolean("FALSE"))
        with self.assertRaises(ValueError):
            parse_boolean("yes")

    def test_valid_request_requires_all_dimensions(self):
        row = {
            "first_turn_id": "record",
            "annotator_id": "annotator",
            "label": "VALID_REQUEST",
            "action_explicit": "true",
            "protected_fields_complete": "true",
            "recipient_or_purpose_scoped": "false",
            "evidence": "May I share it?",
        }
        with self.assertRaisesRegex(ValueError, "all three dimensions"):
            validate_annotation_row(row)

    def test_all_dimensions_require_valid_request_label(self):
        row = {
            "first_turn_id": "record",
            "annotator_id": "annotator",
            "label": "NOT_REQUEST",
            "action_explicit": "true",
            "protected_fields_complete": "true",
            "recipient_or_purpose_scoped": "true",
            "evidence": "May I share it?",
        }
        with self.assertRaisesRegex(ValueError, "require VALID_REQUEST"):
            validate_annotation_row(row)

    def test_kappa_perfect_agreement(self):
        labels = [
            ConsentRequestLabel.VALID_REQUEST,
            ConsentRequestLabel.NOT_REQUEST,
            ConsentRequestLabel.VALID_REQUEST,
        ]
        self.assertEqual(cohen_kappa(labels, labels), 1.0)

    def test_kappa_rejects_different_lengths(self):
        with self.assertRaises(ValueError):
            cohen_kappa(
                [ConsentRequestLabel.VALID_REQUEST],
                [ConsentRequestLabel.VALID_REQUEST, ConsentRequestLabel.NOT_REQUEST],
            )

    def test_confusion_counts_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "equal-length"):
            confusion_counts(
                [ConsentRequestLabel.VALID_REQUEST],
                [],
            )


if __name__ == "__main__":
    unittest.main()
