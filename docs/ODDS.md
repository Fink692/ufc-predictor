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

## Manual Odds Board Command

```powershell
python -m ufc_predictor.cli rank-odds --predictions reports/predictions.csv --odds-board data/odds_board.csv --output reports/value_bets.csv --bankroll 1000
```

Useful options:

- `--bankroll`: bankroll used for stake sizing
- `--kelly-multiplier`: fraction of full Kelly, default `0.25`
- `--max-bankroll-fraction`: maximum stake as bankroll fraction, default `0.02`
- `--min-edge`: minimum model probability edge over implied probability, default `0.02`
- `--min-expected-roi`: minimum expected return per dollar staked, default `0.0`

## Live Odds Fetch

The optional `fetch-odds` command pulls current MMA fight-winner odds from The Odds API and writes the same odds-board schema used by `rank-odds`.

```powershell
$env:THE_ODDS_API_KEY='your_api_key'
python -m ufc_predictor.cli fetch-odds --output data/odds_board.csv --regions us --include-links
```

Defaults:

- sport key: `mma_mixed_martial_arts`
- market: `h2h`
- odds format: American
- API key env var: `THE_ODDS_API_KEY`

You can query exact bookmakers instead of a region:

```powershell
python -m ufc_predictor.cli fetch-odds --bookmakers draftkings,fanduel --include-links --output data/odds_board.csv
```

Use ISO timestamps to limit the event window:

```powershell
python -m ufc_predictor.cli fetch-odds --commence-time-from 2026-08-01T00:00:00Z --commence-time-to 2026-08-02T00:00:00Z
```

Fighter names and `event_date` still need to line up with the upcoming-fights file. The fetcher preserves optional bookmaker, market, outcome, and best available links when the API returns them.

## Full Betting Report

After a model has been trained and an odds board exists, generate predictions, value rankings, and a Markdown summary in one command:

```powershell
python -m ufc_predictor.cli betting-report --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --markdown-output reports/betting_report.md --bankroll 1000
```

The CSV is the detailed machine-readable board. The Markdown file is a quick review sheet for the top value candidates.

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
- `total_payout_if_win`: returned stake plus profit if the bet wins
- `max_loss`: stake amount
- `risk_reward_ratio`: stake divided by potential profit
- `risk_label`: no-bet, lower-variance value, medium-variance value, or high-variance value
- `decision`: bet or pass
- `best_available_for_fighter`: best line found for that fighter across books
- `best_recommendation_for_matchup`: top risk-adjusted qualifying value candidate for that fight

## Risk Framing

"Best" means best risk-adjusted expected value according to this model and the supplied odds. It does not mean safest, guaranteed, or highest payout. Underdogs can have high expected value but also high variance.

The default staking is intentionally conservative:

- quarter Kelly
- capped at 2% of bankroll
- filtered by minimum model edge
