from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.features import FeatureBuilder, build_features_from_raw
from ufc_predictor.io import load_raw_tables


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class FeatureBuilderTests(unittest.TestCase):
    def test_build_features_excludes_no_contests_and_uses_prior_history(self) -> None:
        features = build_features_from_raw(FIXTURES)

        self.assertEqual(len(features), 12)
        self.assertNotIn("ft013", set(features["fight_id"]))

        first = features.loc[features["fight_id"] == "ft001"].iloc[0]
        self.assertEqual(first["fighter_a_ufc_fights"], 0)
        self.assertEqual(first["fighter_b_ufc_fights"], 0)
        self.assertIn("fighter_a_debut_or_missing_history", first["data_quality_flags"])

        later = features.loc[features["fight_id"] == "ft004"].iloc[0]
        self.assertEqual(later["fighter_a_ufc_fights"], 1)
        self.assertEqual(later["fighter_b_ufc_fights"], 1)
        self.assertAlmostEqual(later["fighter_a_strikes_landed_per_15"], 48.0)
        self.assertEqual(later["target_fighter_a_win"], 1)

    def test_prediction_features_match_upcoming_names(self) -> None:
        tables = load_raw_tables(FIXTURES)
        upcoming = pd.read_csv(FIXTURES / "upcoming_fights.csv")
        predictions = FeatureBuilder().build_prediction_frame(tables, upcoming)

        self.assertEqual(len(predictions), 2)
        self.assertNotIn("target_fighter_a_win", predictions.columns)
        self.assertEqual(predictions.iloc[0]["fighter_a"], "Alpha Adams")
        self.assertGreaterEqual(predictions.iloc[0]["fighter_a_ufc_fights"], 4)


if __name__ == "__main__":
    unittest.main()
