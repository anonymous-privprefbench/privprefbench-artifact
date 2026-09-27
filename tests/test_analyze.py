import math
import unittest

from scripts.analyze import ratio, wilson_interval


class AnalyzeTests(unittest.TestCase):
    def test_wilson_interval_for_zero_events_has_nonzero_upper_bound(self):
        lower, upper = wilson_interval(0, 18)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.1759, places=4)

    def test_empty_denominator_is_reported_as_not_applicable(self):
        lower, upper = wilson_interval(0, 0)
        self.assertTrue(math.isnan(lower))
        self.assertTrue(math.isnan(upper))
        self.assertEqual(ratio(0, 0), "n/a (0/0)")


if __name__ == "__main__":
    unittest.main()
