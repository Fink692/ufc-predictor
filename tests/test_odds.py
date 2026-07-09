from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.odds import (
    american_to_implied_probability,
    expected_profit_per_unit,
    kelly_fraction,
    no_vig_probabilities,
    rank_value_bets,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class OddsTests(unittest.TestCase):
    def test_american_odds_conversion(self) -> None:
        self.assertAlmostEqual(american_to_implied_probability(-150), 0.6)
        self.assertAlmostEqual(american_to_implied_probability(130), 1 / 2.3)

    def test_no_vig_probabilities_sum_to_one(self) -> None:
        prob_a, prob_b = no_vig_probabilities(-150, 130)
        self.assertAlmostEqual(prob_a + prob_b, 1.0)
        self.assertGreater(prob_a, prob_b)

    def test_expected_profit_and_kelly(self) -> None:
        decimal = 2.1
        probability = 0.58

        self.assertAlmostEqual(expected_profit_per_unit(probability, decimal), 0.218)
        self.assertAlmostEqual(kelly_fraction(probability, decimal), 0.19818181818181824)

    def test_rank_value_bets_marks_best_lines(self) -> None:
        predictions = pd.read_csv(FIXTURES / "predictions.csv")
        odds_board = pd.read_csv(FIXTURES / "odds_board.csv")

        ranked = rank_value_bets(
            predictions,
            odds_board,
            bankroll=1000,
            kelly_multiplier=0.25,
            max_bankroll_fraction=0.02,
            min_edge=0.02,
        )

        bets = ranked[ranked["decision"] == "bet"]
        self.assertFalse(bets.empty)
        best_alpha = ranked[(ranked["fighter"] == "Alpha Adams") & (ranked["best_available_for_fighter"])]
        self.assertEqual(best_alpha.iloc[0]["sportsbook"], "Book B")
        top = ranked[ranked["best_recommendation_for_matchup"]]
        self.assertGreaterEqual(len(top), 1)
        self.assertTrue((bets["recommended_stake"] <= 20).all())
        self.assertTrue((bets["expected_roi"] > 0).all())


if __name__ == "__main__":
    unittest.main()
