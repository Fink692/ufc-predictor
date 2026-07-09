from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .io import normalize_name


DEFAULT_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_MMA_SPORT_KEY = "mma_mixed_martial_arts"
DEFAULT_MARKET = "h2h"

ODDS_BOARD_COLUMNS = [
    "event_date",
    "fighter_a",
    "fighter_b",
    "sportsbook",
    "fighter",
    "american_odds",
    "commence_time",
    "event_id",
    "bookmaker_key",
    "bookmaker_last_update",
    "market_key",
    "market_last_update",
    "sportsbook_link",
    "market_link",
    "outcome_link",
    "best_link",
    "bookmaker_sid",
    "market_sid",
    "outcome_sid",
    "bet_limit",
]


class OddsApiError(RuntimeError):
    """Raised when live odds cannot be fetched or parsed."""


def fetch_odds_api_events(
    api_key: str,
    sport_key: str = DEFAULT_MMA_SPORT_KEY,
    regions: str = "us",
    markets: str = DEFAULT_MARKET,
    bookmakers: str | None = None,
    commence_time_from: str | None = None,
    commence_time_to: str | None = None,
    include_links: bool = False,
    include_sids: bool = False,
    include_bet_limits: bool = False,
    base_url: str = DEFAULT_ODDS_API_BASE_URL,
    timeout_seconds: int = 30,
) -> list[dict[str, object]]:
    if not api_key:
        raise ValueError("An Odds API key is required. Pass --api-key or set THE_ODDS_API_KEY.")
    if not bookmakers and not regions:
        raise ValueError("Either regions or bookmakers must be provided.")

    params: dict[str, str] = {
        "apiKey": api_key,
        "markets": markets,
        "oddsFormat": "american",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions
    if commence_time_from:
        params["commenceTimeFrom"] = commence_time_from
    if commence_time_to:
        params["commenceTimeTo"] = commence_time_to
    if include_links:
        params["includeLinks"] = "true"
    if include_sids:
        params["includeSids"] = "true"
    if include_bet_limits:
        params["includeBetLimits"] = "true"

    url = f"{base_url.rstrip('/')}/sports/{sport_key}/odds?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "ufc-predictor/0.1"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise OddsApiError(f"The Odds API returned HTTP {error.code}: {body}") from error
    except URLError as error:
        raise OddsApiError(f"Could not reach The Odds API: {error.reason}") from error

    data = json.loads(payload)
    if not isinstance(data, list):
        raise OddsApiError("The Odds API response was not a list of events.")
    return data


def odds_api_events_to_board(events: list[dict[str, object]], market_key: str = DEFAULT_MARKET) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        fighter_a, fighter_b = _event_fighters(event, market_key)
        fighter_names = {normalize_name(fighter_a), normalize_name(fighter_b)} - {""}
        commence_time = str(event.get("commence_time") or "")

        for bookmaker in _as_list(event.get("bookmakers")):
            bookmaker_link = bookmaker.get("link")
            for market in _as_list(bookmaker.get("markets")):
                if market.get("key") != market_key:
                    continue
                for outcome in _as_list(market.get("outcomes")):
                    fighter = str(outcome.get("name") or "").strip()
                    if not fighter or outcome.get("price") is None:
                        continue
                    if fighter_names and normalize_name(fighter) not in fighter_names:
                        continue
                    rows.append(
                        {
                            "event_date": _event_date(commence_time),
                            "fighter_a": fighter_a,
                            "fighter_b": fighter_b,
                            "sportsbook": bookmaker.get("title") or bookmaker.get("key"),
                            "fighter": fighter,
                            "american_odds": outcome.get("price"),
                            "commence_time": commence_time,
                            "event_id": event.get("id"),
                            "bookmaker_key": bookmaker.get("key"),
                            "bookmaker_last_update": bookmaker.get("last_update"),
                            "market_key": market.get("key"),
                            "market_last_update": market.get("last_update"),
                            "sportsbook_link": bookmaker_link,
                            "market_link": market.get("link"),
                            "outcome_link": outcome.get("link"),
                            "best_link": _best_link(outcome.get("link"), market.get("link"), bookmaker_link),
                            "bookmaker_sid": bookmaker.get("sid"),
                            "market_sid": market.get("sid"),
                            "outcome_sid": outcome.get("sid"),
                            "bet_limit": outcome.get("bet_limit"),
                        }
                    )

    return pd.DataFrame(rows, columns=ODDS_BOARD_COLUMNS)


def _event_fighters(event: dict[str, object], market_key: str) -> tuple[str, str]:
    fighter_a = str(event.get("home_team") or "").strip()
    fighter_b = str(event.get("away_team") or "").strip()
    if fighter_a and fighter_b:
        return fighter_a, fighter_b

    for bookmaker in _as_list(event.get("bookmakers")):
        for market in _as_list(bookmaker.get("markets")):
            if market.get("key") != market_key:
                continue
            names = [str(outcome.get("name") or "").strip() for outcome in _as_list(market.get("outcomes"))]
            names = [name for name in names if name]
            if len(names) >= 2:
                return names[0], names[1]
    return fighter_a, fighter_b


def _event_date(commence_time: str) -> str:
    return commence_time[:10] if len(commence_time) >= 10 else ""


def _best_link(*links: object) -> object:
    for link in links:
        if link is not None and str(link).strip():
            return link
    return None


def _as_list(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []
