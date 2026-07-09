from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.odds import american_to_implied_probability, no_vig_probabilities


class OddsTests(unittest.TestCase):
    def test_american_odds_conversion(self) -> None:
        self.assertAlmostEqual(american_to_implied_probability(-150), 0.6)
        self.assertAlmostEqual(american_to_implied_probability(130), 1 / 2.3)

    def test_no_vig_probabilities_sum_to_one(self) -> None:
        prob_a, prob_b = no_vig_probabilities(-150, 130)
        self.assertAlmostEqual(prob_a + prob_b, 1.0)
        self.assertGreater(prob_a, prob_b)


if __name__ == "__main__":
    unittest.main()
