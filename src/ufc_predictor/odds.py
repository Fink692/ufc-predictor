from __future__ import annotations

import math

import pandas as pd


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
