from __future__ import annotations

import math

import pandas as pd

from .io import normalize_name


REQUIRED_ODDS_BOARD_COLUMNS = {
    "event_date",
    "fighter_a",
    "fighter_b",
    "sportsbook",
    "fighter",
    "american_odds",
}

REQUIRED_PREDICTION_COLUMNS = {
    "event_date",
    "fighter_a",
    "fighter_b",
    "prob_fighter_a",
    "prob_fighter_b",
}

OPTIONAL_ODDS_CONTEXT_COLUMNS = (
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
)


def american_to_decimal(price: float | int) -> float:
    price = float(price)
    if price == 0:
        raise ValueError("American odds cannot be zero.")
    if price > 0:
        return 1.0 + price / 100.0
    return 1.0 + 100.0 / abs(price)


def american_to_implied_probability(price: float | int) -> float:
    decimal = american_to_decimal(price)
    return 1.0 / decimal


def expected_profit_per_unit(model_probability: float, decimal_odds: float) -> float:
    return model_probability * decimal_odds - 1.0


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    net_odds = decimal_odds - 1.0
    if net_odds <= 0:
        return 0.0
    fraction = (model_probability * decimal_odds - 1.0) / net_odds
    return max(0.0, fraction)


def no_vig_probabilities(price_a: float | int, price_b: float | int) -> tuple[float, float]:
    prob_a = american_to_implied_probability(price_a)
    prob_b = american_to_implied_probability(price_b)
    total = prob_a + prob_b
    if total <= 0 or math.isnan(total):
        raise ValueError("Odds imply an invalid market total.")
    return prob_a / total, prob_b / total


def add_no_vig_probabilities(
    odds: pd.DataFrame,
    price_a_col: str = "odds_fighter_a",
    price_b_col: str = "odds_fighter_b",
) -> pd.DataFrame:
    enriched = odds.copy()
    probabilities = enriched.apply(
        lambda row: no_vig_probabilities(row[price_a_col], row[price_b_col]),
        axis=1,
        result_type="expand",
    )
    enriched["market_prob_fighter_a"] = probabilities[0]
    enriched["market_prob_fighter_b"] = probabilities[1]
    return enriched


def rank_value_bets(
    predictions: pd.DataFrame,
    odds_board: pd.DataFrame,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
    max_bankroll_fraction: float = 0.02,
    min_edge: float = 0.02,
    min_expected_roi: float = 0.0,
) -> pd.DataFrame:
    _validate_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "predictions")
    _validate_columns(odds_board, REQUIRED_ODDS_BOARD_COLUMNS, "odds board")
    if bankroll <= 0:
        raise ValueError("bankroll must be positive.")
    if kelly_multiplier < 0:
        raise ValueError("kelly_multiplier cannot be negative.")
    if max_bankroll_fraction < 0:
        raise ValueError("max_bankroll_fraction cannot be negative.")

    prediction_lookup = _prediction_lookup(predictions)
    rows: list[dict[str, object]] = []
    for raw in odds_board.to_dict("records"):
        key = _matchup_key(raw)
        if key not in prediction_lookup:
            raise ValueError(
                "Odds row did not match any prediction: "
                f"{raw.get('event_date')} {raw.get('fighter_a')} vs. {raw.get('fighter_b')}"
            )
        prediction = prediction_lookup[key]
        fighter_name = str(raw["fighter"]).strip()
        fighter_key = normalize_name(fighter_name)
        fighter_a_key = normalize_name(prediction["fighter_a"])
        fighter_b_key = normalize_name(prediction["fighter_b"])
        if fighter_key == fighter_a_key:
            model_probability = float(prediction["prob_fighter_a"])
            opponent = prediction["fighter_b"]
            side = "fighter_a"
        elif fighter_key == fighter_b_key:
            model_probability = float(prediction["prob_fighter_b"])
            opponent = prediction["fighter_a"]
            side = "fighter_b"
        else:
            raise ValueError(
                f"Odds fighter {fighter_name!r} is not part of matchup "
                f"{prediction['fighter_a']} vs. {prediction['fighter_b']}."
            )

        american_odds = float(raw["american_odds"])
        decimal_odds = american_to_decimal(american_odds)
        implied_probability = american_to_implied_probability(american_odds)
        edge = model_probability - implied_probability
        expected_roi = expected_profit_per_unit(model_probability, decimal_odds)
        full_kelly = kelly_fraction(model_probability, decimal_odds)
        recommended_fraction = min(full_kelly * kelly_multiplier, max_bankroll_fraction)
        if edge < min_edge or expected_roi < min_expected_roi:
            recommended_fraction = 0.0
        recommended_stake = bankroll * recommended_fraction
        volatility = _profit_volatility(model_probability, decimal_odds, expected_roi)
        risk_adjusted_score = expected_roi / volatility if volatility > 0 else 0.0
        potential_profit = recommended_stake * (decimal_odds - 1.0)
        output_row = {
            "event_date": prediction["event_date"],
            "fighter_a": prediction["fighter_a"],
            "fighter_b": prediction["fighter_b"],
            "sportsbook": raw["sportsbook"],
            "fighter": fighter_name,
            "opponent": opponent,
            "side": side,
            "american_odds": int(american_odds) if american_odds.is_integer() else american_odds,
            "decimal_odds": decimal_odds,
            "model_probability": model_probability,
            "implied_probability": implied_probability,
            "edge": edge,
            "expected_roi": expected_roi,
            "expected_profit_per_100": expected_roi * 100.0,
            "full_kelly_fraction": full_kelly,
            "recommended_fraction": recommended_fraction,
            "recommended_stake": recommended_stake,
            "potential_profit": potential_profit,
            "total_payout_if_win": recommended_stake * decimal_odds,
            "max_loss": recommended_stake,
            "risk_reward_ratio": recommended_stake / potential_profit if potential_profit > 0 else None,
            "profit_volatility_per_1": volatility,
            "risk_adjusted_score": risk_adjusted_score,
            "risk_label": _risk_label(model_probability, decimal_odds, expected_roi, recommended_fraction),
            "decision": "bet" if recommended_stake > 0 else "pass",
        }
        for column in OPTIONAL_ODDS_CONTEXT_COLUMNS:
            if column in raw:
                output_row[column] = raw[column]
        rows.append(output_row)

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked["best_available_for_fighter"] = (
        ranked.groupby(["event_date", "fighter_a", "fighter_b", "fighter"])["expected_roi"]
        .rank(method="first", ascending=False)
        .eq(1)
    )
    ranked["best_recommendation_for_matchup"] = False
    bet_rows = ranked[ranked["decision"] == "bet"]
    if not bet_rows.empty:
        best_indexes = bet_rows.groupby(["event_date", "fighter_a", "fighter_b"])["risk_adjusted_score"].idxmax()
        ranked.loc[best_indexes, "best_recommendation_for_matchup"] = True
    ranked = ranked.sort_values(
        ["decision", "risk_adjusted_score", "expected_roi", "edge"],
        ascending=[True, False, False, False],
    )
    return ranked.reset_index(drop=True)


def value_bets_to_markdown(value_bets: pd.DataFrame, top_n: int = 10, bankroll: float | None = None) -> str:
    if value_bets.empty:
        return "\n".join(
            [
                "# UFC Betting Value Report",
                "",
                "Analytics only. This is not betting advice.",
                "",
                "No odds rows were available to rank.",
                "",
            ]
        )

    bets = value_bets[value_bets["decision"] == "bet"].head(top_n)
    generated = value_bets.head(top_n) if bets.empty else bets
    lines = [
        "# UFC Betting Value Report",
        "",
        "Analytics only. This is not betting advice.",
        "",
    ]
    if bankroll is not None:
        lines.append(f"Bankroll used for sizing: ${bankroll:,.2f}")
        lines.append("")

    bet_count = int((value_bets["decision"] == "bet").sum())
    matchup_count = int(value_bets[["event_date", "fighter_a", "fighter_b"]].drop_duplicates().shape[0])
    lines.extend(
        [
            f"Ranked {len(value_bets)} sportsbook lines across {matchup_count} matchups.",
            f"Qualifying value candidates: {bet_count}.",
            "",
        ]
    )

    if bets.empty:
        lines.extend(
            [
                "No rows passed the configured edge, expected-return, and stake filters.",
                "",
                "The table below shows the highest-ranked lines anyway for review.",
                "",
            ]
        )

    lines.extend(
        [
            "| Matchup | Fighter | Sportsbook | Odds | Model | Implied | Edge | EV/$ | Stake | Profit | Risk | Link |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for _, row in generated.iterrows():
        matchup = f"{row['fighter_a']} vs. {row['fighter_b']}"
        link = _markdown_link(row)
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(matchup),
                    _markdown_cell(row["fighter"]),
                    _markdown_cell(row["sportsbook"]),
                    _format_american(row["american_odds"]),
                    _format_percent(row["model_probability"]),
                    _format_percent(row["implied_probability"]),
                    _format_percent(row["edge"]),
                    _format_percent(row["expected_roi"]),
                    _format_money(row["recommended_stake"]),
                    _format_money(row["potential_profit"]),
                    _markdown_cell(row["risk_label"]),
                    link,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Recommended stake is conservative Kelly sizing after caps and filters. Max loss is the stake.",
            "",
        ]
    )
    return "\n".join(lines)


def _prediction_lookup(predictions: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in predictions.to_dict("records"):
        lookup[_matchup_key(row)] = row
    return lookup


def _matchup_key(row: dict[str, object]) -> tuple[str, str, str]:
    fighters = sorted([normalize_name(row["fighter_a"]), normalize_name(row["fighter_b"])])
    return str(row["event_date"]), fighters[0], fighters[1]


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} CSV missing columns: {missing}")


def _profit_volatility(model_probability: float, decimal_odds: float, expected_roi: float) -> float:
    win_profit = decimal_odds - 1.0
    loss_profit = -1.0
    variance = model_probability * (win_profit - expected_roi) ** 2
    variance += (1.0 - model_probability) * (loss_profit - expected_roi) ** 2
    return math.sqrt(max(variance, 0.0))


def _risk_label(
    model_probability: float,
    decimal_odds: float,
    expected_roi: float,
    recommended_fraction: float,
) -> str:
    if recommended_fraction <= 0 or expected_roi <= 0:
        return "no_bet"
    if model_probability >= 0.62 and decimal_odds <= 2.2 and recommended_fraction <= 0.01:
        return "lower_variance_value"
    if model_probability >= 0.52 and decimal_odds <= 3.0:
        return "medium_variance_value"
    return "high_variance_value"


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def _format_money(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"${float(value):,.2f}"


def _format_american(value: object) -> str:
    if pd.isna(value):
        return ""
    price = float(value)
    rounded = int(price) if price.is_integer() else price
    if price > 0:
        return f"+{rounded}"
    return str(rounded)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "/")


def _markdown_link(row: pd.Series) -> str:
    for column in ["best_link", "outcome_link", "market_link", "sportsbook_link"]:
        if column in row and not pd.isna(row[column]) and str(row[column]).strip():
            return f"[open]({row[column]})"
    return ""
