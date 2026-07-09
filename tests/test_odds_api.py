from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ufc_predictor.odds_api import fetch_odds_api_events, odds_api_events_to_board


SAMPLE_EVENTS = [
    {
        "id": "event-1",
        "sport_key": "mma_mixed_martial_arts",
        "commence_time": "2026-08-01T23:00:00Z",
        "home_team": "Alpha Adams",
        "away_team": "Ethan Ellis",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "last_update": "2026-07-09T12:00:00Z",
                "link": "https://sportsbook.example/event",
                "sid": "book-1",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-07-09T12:00:00Z",
                        "link": "https://sportsbook.example/market",
                        "sid": "market-1",
                        "outcomes": [
                            {
                                "name": "Alpha Adams",
                                "price": -125,
                                "link": "https://sportsbook.example/alpha",
                                "sid": "outcome-1",
                                "bet_limit": 500,
                            },
                            {"name": "Ethan Ellis", "price": 115},
                            {"name": "Draw", "price": 5000},
                        ],
                    }
                ],
            }
        ],
    }
]


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class OddsApiTests(unittest.TestCase):
    def test_odds_api_events_to_board_outputs_rankable_moneyline_rows(self) -> None:
        board = odds_api_events_to_board(SAMPLE_EVENTS)

        self.assertEqual(len(board), 2)
        self.assertEqual(board.iloc[0]["event_date"], "2026-08-01")
        self.assertEqual(board.iloc[0]["fighter_a"], "Alpha Adams")
        self.assertEqual(board.iloc[0]["fighter_b"], "Ethan Ellis")
        self.assertEqual(board.iloc[0]["sportsbook"], "DraftKings")
        self.assertEqual(board.iloc[0]["american_odds"], -125)
        self.assertEqual(board.iloc[0]["best_link"], "https://sportsbook.example/alpha")
        self.assertEqual(board.iloc[0]["bet_limit"], 500)
        self.assertNotIn("Draw", set(board["fighter"]))

    def test_fetch_odds_api_events_builds_v4_mma_request(self) -> None:
        with patch("ufc_predictor.odds_api.urlopen", return_value=FakeResponse(b"[]")) as mocked_urlopen:
            events = fetch_odds_api_events(
                api_key="test-key",
                bookmakers="draftkings,fanduel",
                include_links=True,
                include_sids=True,
                commence_time_from="2026-08-01T00:00:00Z",
                commence_time_to="2026-08-02T00:00:00Z",
            )

        self.assertEqual(events, [])
        request = mocked_urlopen.call_args.args[0]
        parsed = urlparse(request.full_url)
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/v4/sports/mma_mixed_martial_arts/odds")
        self.assertEqual(query["apiKey"], ["test-key"])
        self.assertEqual(query["bookmakers"], ["draftkings,fanduel"])
        self.assertNotIn("regions", query)
        self.assertEqual(query["markets"], ["h2h"])
        self.assertEqual(query["oddsFormat"], ["american"])
        self.assertEqual(query["includeLinks"], ["true"])
        self.assertEqual(query["includeSids"], ["true"])
        self.assertEqual(query["commenceTimeFrom"], ["2026-08-01T00:00:00Z"])
        self.assertEqual(query["commenceTimeTo"], ["2026-08-02T00:00:00Z"])


if __name__ == "__main__":
    unittest.main()
