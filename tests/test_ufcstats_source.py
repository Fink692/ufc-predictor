from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.ufcstats_source import _convert_fight_stats


class UfcStatsSourceTests(unittest.TestCase):
    def test_late_and_championship_round_stats_are_aggregated(self) -> None:
        fights = pd.DataFrame(
            [
                {
                    "fight_id": "fight-1",
                    "source_event": "UFC Test",
                    "source_bout": "Alpha Adams vs. Bruno Blake",
                    "finish_round": 4,
                    "finish_time": "2:30",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                _round("Round 1", "Alpha Adams", "1 of 2", "2 of 3", "0 of 1", "0:10"),
                _round("Round 2", "Alpha Adams", "3 of 4", "4 of 5", "1 of 2", "0:20"),
                _round("Round 3", "Alpha Adams", "5 of 6", "6 of 7", "1 of 1", "0:30"),
                _round("Round 4", "Alpha Adams", "7 of 8", "8 of 9", "0 of 0", "0:40"),
                _round("Round 1", "Bruno Blake", "2 of 3", "3 of 4", "0 of 0", "0:05"),
                _round("Round 3", "Bruno Blake", "4 of 5", "5 of 6", "1 of 2", "0:15"),
                _round("Round 4", "Bruno Blake", "6 of 7", "7 of 8", "0 of 1", "0:25"),
            ]
        )

        converted = _convert_fight_stats(stats, fights, {"alpha adams": "a", "bruno blake": "b"})
        alpha = converted[converted["fighter_id"] == "a"].iloc[0]

        self.assertEqual(alpha["strikes_succ"], 16)
        self.assertEqual(alpha["late_strikes_succ"], 12)
        self.assertEqual(alpha["champ_strikes_succ"], 7)
        self.assertEqual(alpha["late_takedown_succ"], 1)
        self.assertEqual(alpha["late_ctrl_time_seconds"], 70)
        self.assertAlmostEqual(alpha["fight_minutes"], 17.5)
        self.assertAlmostEqual(alpha["late_fight_minutes"], 7.5)
        self.assertAlmostEqual(alpha["champ_fight_minutes"], 2.5)


def _round(
    round_name: str,
    fighter: str,
    sig_str: str,
    total_str: str,
    takedown: str,
    control: str,
) -> dict[str, object]:
    return {
        "EVENT": "UFC Test",
        "BOUT": "Alpha Adams vs. Bruno Blake",
        "ROUND": round_name,
        "FIGHTER": fighter,
        "KD": 0,
        "SIG.STR.": sig_str,
        "SIG.STR. %": "50%",
        "TOTAL STR.": total_str,
        "TD": takedown,
        "TD %": "50%",
        "SUB.ATT": 0,
        "REV.": 0,
        "CTRL": control,
        "HEAD": "0 of 0",
        "BODY": "0 of 0",
        "LEG": "0 of 0",
        "DISTANCE": "0 of 0",
        "CLINCH": "0 of 0",
        "GROUND": "0 of 0",
    }


if __name__ == "__main__":
    unittest.main()
