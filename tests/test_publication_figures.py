from __future__ import annotations

import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_publication_figures import figure_one, figure_two, load_rows


class PublicationFigureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_rows(
            ROOT / "analysis/publication-v3-human-gold/first-turn-records.csv"
        )

    def test_figure_one_is_valid_svg_with_observed_extremum(self):
        svg = figure_one(self.rows)
        ET.fromstring(svg)
        self.assertIn("56.7%", svg)
        self.assertIn("(34/60)", svg)

    def test_figure_two_is_valid_svg_with_provider_coordinates(self):
        svg = figure_two(self.rows)
        ET.fromstring(svg)
        self.assertIn("(77.2%, 91.7%)", svg)
        self.assertIn("(96.1%, 72.8%)", svg)


if __name__ == "__main__":
    unittest.main()
