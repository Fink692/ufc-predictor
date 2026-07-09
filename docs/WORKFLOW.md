# Workflow Guide

This guide keeps the runnable commands outside the front-page README.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

After installation, `ufc-predict` is available as the CLI entry point.

## Train From Public Data

```powershell
ufc-predict download-ufcstats --output-raw-dir data/raw
ufc-predict build-features --raw-dir data/raw --output data/processed/features.csv
ufc-predict train --features data/processed/features.csv --model-path models/ufc_model.joblib --report reports/train_metrics.json
```

## Compare Models And Calibration

```powershell
ufc-predict compare-models --features data/processed/features.csv --output reports/model_comparison.csv
ufc-predict calibration-report --features data/processed/features.csv --model-path models/ufc_model.joblib --output reports/calibration.csv --report reports/calibration_summary.json
```

The comparison command evaluates regularized logistic regression, a less-regularized logistic baseline, random forest, and histogram gradient boosting on the same chronological holdout. The calibration command writes probability buckets so you can see whether model confidence matches observed outcomes.

## Score Upcoming Fights

Create `data/upcoming_fights.csv` with:

```csv
event_date,fighter_a,fighter_b,weight_class,gender,scheduled_rounds,title_fight
2026-06-01,Fighter One,Fighter Two,Welterweight,M,5,true
```

Then score the card:

```powershell
ufc-predict predict --raw-dir data/raw --model-path models/ufc_model.joblib --input data/upcoming_fights.csv --output reports/predictions.csv
```

## Rank Manual Odds

Create `data/odds_board.csv` with one row per sportsbook line:

```csv
event_date,fighter_a,fighter_b,sportsbook,fighter,american_odds
2026-06-01,Fighter One,Fighter Two,Book A,Fighter One,-125
2026-06-01,Fighter One,Fighter Two,Book B,Fighter One,+105
2026-06-01,Fighter One,Fighter Two,Book A,Fighter Two,+115
```

Rank the board:

```powershell
ufc-predict rank-odds --predictions reports/predictions.csv --odds-board data/odds_board.csv --output reports/value_bets.csv --bankroll 1000
```

## Build A Betting Report

```powershell
ufc-predict betting-report --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --fight-output reports/fight_recommendations.csv --markdown-output reports/betting_report.md --workbook-output reports/betting_report.xlsx --bankroll 1000 --max-confidence-stake 100
```

The report includes:

- `reports/predictions.csv`
- `reports/value_bets.csv`
- `reports/fight_recommendations.csv`
- `reports/betting_report.md`
- `reports/betting_report.xlsx`

## Fetch Live Odds

Live odds use [The Odds API](https://the-odds-api.com/sports/mma-ufc-odds.html).

```powershell
$env:THE_ODDS_API_KEY='your_api_key'
ufc-predict betting-report --fetch-live-odds --include-links --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --fight-output reports/fight_recommendations.csv --markdown-output reports/betting_report.md --workbook-output reports/betting_report.xlsx --bankroll 1000 --max-confidence-stake 100
```

Use `--bookmakers draftkings,fanduel` to query specific books instead of the default US region.

## Backtest Historical Sportsbook Odds

Historical odds backtesting requires historical model predictions with actual winners plus a historical odds board.

Prediction columns:

- `event_date`
- `fighter_a`
- `fighter_b`
- `prob_fighter_a`
- `prob_fighter_b`
- `actual_winner` or `target_fighter_a_win`

Historical odds columns use the same odds-board schema:

- `event_date`
- `fighter_a`
- `fighter_b`
- `sportsbook`
- `fighter`
- `american_odds`

Run the replay:

```powershell
ufc-predict historical-odds-backtest --predictions reports/historical_predictions.csv --odds-board data/historical_odds_board.csv --output reports/historical_odds_backtest.csv --summary-output reports/historical_odds_strategy_summary.csv --workbook-output reports/historical_odds_backtest.xlsx --bankroll 1000 --flat-stake 10 --max-confidence-stake 100
```

The output compares conservative Kelly, flat-stake, and confidence-stake strategies with total stake, net profit, ROI, ending bankroll, and max drawdown.

## Build The Holdout Workbook

```powershell
ufc-predict backtest-workbook --features data/processed/features.csv --output reports/ufc_backtest_tables_charts.xlsx --rows-output reports/ufc_holdout_backtest_rows.csv
```

## Optional Context Variables

The raw `fights.csv` and upcoming fight CSV can include these extra columns when you have reliable sources:

- `fighter_a_short_notice`, `fighter_b_short_notice`
- `fighter_a_weight_miss`, `fighter_b_weight_miss`
- `fighter_a_camp_change`, `fighter_b_camp_change`
- `fighter_a_disclosed_injury`, `fighter_b_disclosed_injury`
- `fighter_a_camp`, `fighter_b_camp`
- `altitude_ft`
- `fighter_a_travel_distance_km`, `fighter_b_travel_distance_km`, or `travel_distance_diff_km`

Missing values default to neutral values. Do not backfill these columns with information that was only known after the fight.

## Dashboard

The dashboard is optional and reads generated prediction/odds CSVs.

```powershell
python -m pip install -e ".[dashboard]"
streamlit run dashboard/streamlit_app.py
```

## Test

```powershell
python -m unittest discover -v
python -m compileall -q src tests
```
