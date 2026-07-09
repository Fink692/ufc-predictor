from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import json

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.cli import main


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class CliFlowTests(unittest.TestCase):
    def test_train_evaluate_predict_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "features.csv"
            model = root / "models" / "ufc_model.joblib"
            train_report = root / "reports" / "train_metrics.json"
            eval_report = root / "reports" / "evaluation.json"
            predictions = root / "reports" / "predictions.csv"
            value_bets = root / "reports" / "value_bets.csv"

            main(["build-features", "--raw-dir", str(FIXTURES), "--output", str(features)])
            main(["train", "--features", str(features), "--model-path", str(model), "--report", str(train_report)])
            main(
                [
                    "evaluate",
                    "--features",
                    str(features),
                    "--model-path",
                    str(model),
                    "--report",
                    str(eval_report),
                    "--odds",
                    str(FIXTURES / "odds.csv"),
                ]
            )
            main(
                [
                    "predict",
                    "--raw-dir",
                    str(FIXTURES),
                    "--model-path",
                    str(model),
                    "--input",
                    str(FIXTURES / "upcoming_fights.csv"),
                    "--output",
                    str(predictions),
                ]
            )
            main(
                [
                    "rank-odds",
                    "--predictions",
                    str(FIXTURES / "predictions.csv"),
                    "--odds-board",
                    str(FIXTURES / "odds_board.csv"),
                    "--output",
                    str(value_bets),
                    "--bankroll",
                    "1000",
                ]
            )

            self.assertTrue(model.exists())
            self.assertTrue(train_report.exists())
            self.assertTrue(eval_report.exists())
            report = json.loads(eval_report.read_text(encoding="utf-8"))
            self.assertIn("walk_forward_event_time", report)
            self.assertIsNotNone(report["walk_forward_event_time"]["metrics"])
            self.assertIn("market_odds_baseline", report)
            self.assertEqual(report["market_odds_baseline"]["coverage_rows"], 12)
            output = pd.read_csv(predictions)
            self.assertEqual(len(output), 2)
            self.assertIn("prob_fighter_a", output.columns)
            self.assertTrue(output["prob_fighter_a"].between(0, 1).all())
            self.assertTrue(output["prob_fighter_b"].between(0, 1).all())
            odds_output = pd.read_csv(value_bets)
            self.assertEqual(len(odds_output), 8)
            self.assertIn("expected_roi", odds_output.columns)
            self.assertTrue((odds_output["decision"] == "bet").any())


if __name__ == "__main__":
    unittest.main()
