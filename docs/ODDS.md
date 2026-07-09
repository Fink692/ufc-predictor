# Odds And Risk

The `rank-odds` command turns model probabilities and sportsbook lines into a ranked value-bet board.

It does not guarantee profit. It is an analytics tool for comparing lines.

## Input

`reports/predictions.csv` from the `predict` command must include:

- `event_date`
- `fighter_a`
- `fighter_b`
- `prob_fighter_a`
- `prob_fighter_b`

`data/odds_board.csv` must include:

- `event_date`
- `fighter_a`
- `fighter_b`
- `sportsbook`
- `fighter`
- `american_odds`

Each sportsbook line is one row. Include both fighters and as many sportsbooks as you want to compare.

## Command

```powershell
python -m ufc_predictor.cli rank-odds --predictions reports/predictions.csv --odds-board data/odds_board.csv --output reports/value_bets.csv --bankroll 1000
```

Useful options:

- `--bankroll`: bankroll used for stake sizing
- `--kelly-multiplier`: fraction of full Kelly, default `0.25`
- `--max-bankroll-fraction`: maximum stake as bankroll fraction, default `0.02`
- `--min-edge`: minimum model probability edge over implied probability, default `0.02`
- `--min-expected-roi`: minimum expected return per dollar staked, default `0.0`

## Output

Important columns:

- `model_probability`: model win probability for that fighter
- `implied_probability`: sportsbook break-even probability
- `edge`: `model_probability - implied_probability`
- `expected_roi`: expected profit per $1 staked
- `expected_profit_per_100`: expected profit per $100 staked
- `full_kelly_fraction`: full Kelly stake fraction
- `recommended_fraction`: conservative capped Kelly fraction
- `recommended_stake`: suggested stake for the provided bankroll
- `potential_profit`: profit if the bet wins
- `max_loss`: stake amount
- `risk_label`: no-bet, lower-variance value, medium-variance value, or high-variance value
- `decision`: bet or pass

## Risk Framing

"Best" means best risk-adjusted expected value according to this model and the supplied odds. It does not mean safest, guaranteed, or highest payout. Underdogs can have high expected value but also high variance.

The default staking is intentionally conservative:

- quarter Kelly
- capped at 2% of bankroll
- filtered by minimum model edge
