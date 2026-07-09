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

## Build The Holdout Workbook

```powershell
ufc-predict backtest-workbook --features data/processed/features.csv --output reports/ufc_backtest_tables_charts.xlsx --rows-output reports/ufc_holdout_backtest_rows.csv
```

## Test

```powershell
python -m unittest discover -v
python -m compileall -q src tests
```
