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
python -m ufc_predictor.cli betting-report --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --fight-output reports/fight_recommendations.csv --markdown-output reports/betting_report.md --workbook-output reports/betting_report.xlsx --bankroll 1000 --max-confidence-stake 100
```

To fetch live odds first and then generate the same reports:

```powershell
$env:THE_ODDS_API_KEY='your_api_key'
python -m ufc_predictor.cli betting-report --fetch-live-odds --include-links --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --fight-output reports/fight_recommendations.csv --markdown-output reports/betting_report.md --workbook-output reports/betting_report.xlsx --bankroll 1000 --max-confidence-stake 100
```

The `--output` CSV is the detailed sportsbook-line board. The command also writes a per-fight confidence-stake file to `reports/fight_recommendations.csv` by default. The Markdown file includes both the top value candidates and the fight-by-fight confidence bets. Add `--workbook-output reports/betting_report.xlsx` to create a full Excel workbook with all tables and charts.

Useful options:

- `--fetch-live-odds`: fetch The Odds API lines into `--odds-board` before building the reports
- `--fight-output`: per-fight recommendation CSV path
- `--workbook-output`: optional Excel workbook with summary, recommendations, value board, all odds, predictions, and charts
- `--max-confidence-stake`: stake used at 100% adjusted confidence, default `100`
- `--top-n`: number of rows to include in the Markdown sections

## Output

Important value-board columns:

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

Important per-fight recommendation columns:

- `predicted_winner`: model pick for the fight
- `predicted_win_probability`: raw model win probability for that pick
- `confidence`: coin-flip-adjusted confidence, where 50% win probability is 0 confidence and 100% is 1
- `confidence_stake`: stake from `confidence * max_confidence_stake`
- `best_sportsbook`: book with the best available odds for the predicted winner
- `best_american_odds`: best American odds for the predicted winner
- `profit_if_correct`: profit from the confidence stake if the pick wins
- `total_return_if_correct`: stake plus profit if the pick wins
- `max_loss_if_wrong`: confidence stake
- `expected_profit`: model expected profit from that stake and line
- `value_flag`: `positive_ev`, `negative_ev`, or `missing_odds`
- `all_predicted_winner_odds`: every available line found for the pick

## Historical Odds Backtest

If you have real historical moneylines, the `historical-odds-backtest` command replays the value-board logic against actual fight outcomes.

```powershell
ufc-predict historical-odds-backtest --predictions reports/historical_predictions.csv --odds-board data/historical_odds_board.csv --output reports/historical_odds_backtest.csv --summary-output reports/historical_odds_strategy_summary.csv --workbook-output reports/historical_odds_backtest.xlsx --bankroll 1000 --flat-stake 10 --max-confidence-stake 100
```

The historical prediction file must include `actual_winner` or `target_fighter_a_win`. The odds board uses the same row-per-bookmaker-line schema as live odds.

The backtest compares:

- conservative Kelly stake from the value board
- fixed flat stake
- confidence stake where model confidence maps to dollars

The strategy summary reports bet count, wins, hit rate, total stake, net profit, ROI, ending bankroll, and max drawdown.

Workbook sheets:

- `Summary`: report-level assumptions, counts, total confidence stake, expected profit, and worst-case confidence-stake loss
- `Fight Recommendations`: one row per fight with the model pick, best odds, confidence stake, profit if correct, and expected profit
- `Value Board`: every sportsbook line ranked by edge, expected ROI, conservative Kelly stake, payout, max loss, and risk label
- `Best Lines`: the best available line for each fighter across books
- `Top Matchups`: the best risk-adjusted value candidate per matchup
- `Predictions`: raw model probabilities and confidence
- `Odds Board`: the supplied or fetched odds input
- `Charts`: stake, expected-profit, edge, value-candidate, and risk-label charts

## Risk Framing

"Best" means best risk-adjusted expected value according to this model and the supplied odds. It does not mean safest, guaranteed, or highest payout. Underdogs can have high expected value but also high variance.

The confidence-stake view is intentionally simple:

```text
confidence = (predicted_win_probability - 0.50) * 2
confidence_stake = confidence * max_confidence_stake
```

That means a 58% model pick becomes 16% adjusted confidence and a `$16` stake when `--max-confidence-stake 100`. This shows what the confidence scaling would do; it does not mean every confidence-sized stake is positive expected value.

The default staking is intentionally conservative:

- quarter Kelly
- capped at 2% of bankroll
- filtered by minimum model edge
