from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.odds import (
    american_to_implied_probability,
    build_fight_recommendations,
    expected_profit_per_unit,
    fight_recommendations_to_markdown,
    kelly_fraction,
    no_vig_probabilities,
    rank_value_bets,
    value_bets_to_markdown,
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
        self.assertIn("total_payout_if_win", ranked.columns)
        self.assertIn("risk_reward_ratio", ranked.columns)

    def test_value_bets_to_markdown_summarizes_top_candidates(self) -> None:
        predictions = pd.read_csv(FIXTURES / "predictions.csv")
        odds_board = pd.read_csv(FIXTURES / "odds_board.csv")
        ranked = rank_value_bets(predictions, odds_board, bankroll=1000)

        report = value_bets_to_markdown(ranked, top_n=3, bankroll=1000)

        self.assertIn("# UFC Betting Value Report", report)
        self.assertIn("Analytics only. This is not betting advice.", report)
        self.assertIn("| Matchup | Fighter | Sportsbook | Odds |", report)
        self.assertIn("Qualifying value candidates:", report)

    def test_build_fight_recommendations_sizes_stake_by_confidence(self) -> None:
        predictions = pd.read_csv(FIXTURES / "predictions.csv")
        odds_board = pd.read_csv(FIXTURES / "odds_board.csv")

        recommendations = build_fight_recommendations(predictions, odds_board, max_confidence_stake=100)

        self.assertEqual(len(recommendations), 2)
        top = recommendations.iloc[0]
        self.assertEqual(top["predicted_winner"], "Alpha Adams")
        self.assertEqual(top["best_sportsbook"], "Book B")
        self.assertEqual(top["best_american_odds"], 105)
        self.assertAlmostEqual(top["predicted_win_probability"], 0.58)
        self.assertAlmostEqual(top["confidence"], 0.16)
        self.assertAlmostEqual(top["confidence_stake"], 16.0)
        self.assertAlmostEqual(top["profit_if_correct"], 16.8)
        self.assertIn("Book A -125", top["all_predicted_winner_odds"])
        self.assertIn("Book B +105", top["all_predicted_winner_odds"])

    def test_fight_recommendations_to_markdown_lists_confidence_bets(self) -> None:
        predictions = pd.read_csv(FIXTURES / "predictions.csv")
        odds_board = pd.read_csv(FIXTURES / "odds_board.csv")
        recommendations = build_fight_recommendations(predictions, odds_board, max_confidence_stake=100)

        report = fight_recommendations_to_markdown(recommendations)

        self.assertIn("# Fight Confidence Bets", report)
        self.assertIn("0% confidence = $0", report)
        self.assertIn("Alpha Adams", report)
        self.assertIn("$16.00", report)


if __name__ == "__main__":
    unittest.main()
