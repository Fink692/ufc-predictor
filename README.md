# UFC Predictor

Leakage-safe UFC fight winner prediction pipeline built for reproducible MMA analytics.

This project trains winner models from historical UFCStats-style fight data using only information available before each bout. It includes public-data ingestion, rolling feature generation, model training, chronological evaluation, and upcoming-card scoring.

> Analytics only. This is not betting advice.

## Highlights

- Public UFCStats mirror ingestion from [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats)
- Rolling pre-fight features to avoid future-data leakage
- Age, reach, stance, layoffs, Elo, recent form, opponent strength, grappling, striking, control, and stamina/fade features
- Late-round stamina features from round 3+ and championship-round performance
- Order-balanced training so the model does not simply learn UFCStats fighter ordering
- Sportsbook line ranking by expected value, edge, conservative Kelly sizing, and risk labels
- Optional live MMA odds ingestion from The Odds API
- One-command betting report generation with CSV and Markdown outputs
- Chronological holdout reporting with accuracy, log loss, Brier score, ROC AUC, and baselines
- CLI-first workflow that can be rerun from raw public data

## Current Large-Data Benchmark

Using the public UFCStats mirror available during the latest local run:

| Item | Value |
| --- | ---: |
| Source fights | 8,701 |
| Trainable fights | 8,547 |
| Date range | 1994-03-11 to 2026-05-16 |
| Model features | 148 |
| Stamina/fade features | 39 |
| Holdout split | latest chronological 20% |
| Holdout rows | 1,709 |

| Metric | Model | Baseline |
| --- | ---: | ---: |
| Accuracy | 63.55% | 55.71% 50/50/fighter-A baseline |
| Log loss | 0.6441 | 0.6931 50/50 baseline |
| ROC AUC | 0.6695 | 0.5000 50/50 baseline |

The latest generated local report is written to `reports/large_training_summary.json`, which is ignored by git because it is reproducible output.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```powershell
python -m pip install -e ".[dev]"
```

## Reproduce The Pipeline

Run from the repository root:

```powershell
python -m ufc_predictor.cli download-ufcstats --output-raw-dir data/raw
python -m ufc_predictor.cli build-features --raw-dir data/raw --output data/processed/features.csv
python -m ufc_predictor.cli train --features data/processed/features.csv --model-path models/ufc_model.joblib --report reports/train_metrics.json
```

To score upcoming fights after a model is trained:

```powershell
python -m ufc_predictor.cli predict --raw-dir data/raw --model-path models/ufc_model.joblib --input data/upcoming_fights.csv --output reports/predictions.csv
```

To rank sportsbook odds after predictions are generated:

```powershell
python -m ufc_predictor.cli rank-odds --predictions reports/predictions.csv --odds-board data/odds_board.csv --output reports/value_bets.csv --bankroll 1000
```

To generate the complete betting report from upcoming fights and an odds board:

```powershell
python -m ufc_predictor.cli betting-report --raw-dir data/raw --model-path models/ufc_model.joblib --upcoming data/upcoming_fights.csv --odds-board data/odds_board.csv --predictions-output reports/predictions.csv --output reports/value_bets.csv --markdown-output reports/betting_report.md --bankroll 1000
```

After editable install, use `ufc-predict` instead of `python -m ufc_predictor.cli`.

## Upcoming Fight Input

`data/upcoming_fights.csv`

```csv
event_date,fighter_a,fighter_b,weight_class,gender,scheduled_rounds,title_fight
2026-06-01,Fighter One,Fighter Two,Welterweight,M,5,true
```

Fighter names are matched after normalization against `fighters.csv`.

## Odds Board Input

`data/odds_board.csv`

```csv
event_date,fighter_a,fighter_b,sportsbook,fighter,american_odds
2026-06-01,Fighter One,Fighter Two,Book A,Fighter One,-125
2026-06-01,Fighter One,Fighter Two,Book B,Fighter One,+105
2026-06-01,Fighter One,Fighter Two,Book A,Fighter Two,+115
```

The `rank-odds` command compares each sportsbook line against the model probability and outputs implied probability, model edge, expected return, conservative Kelly stake sizing, potential profit, max loss, risk label, and bet/pass decision.

The default stake sizing uses quarter Kelly capped at 2% of bankroll. This is intentionally conservative and still does not make any bet safe.

## Live Odds Fetching

The project can optionally fetch current MMA fight-winner odds from [The Odds API](https://the-odds-api.com/sports/mma-ufc-odds.html). Set an API key and write an odds board:

```powershell
$env:THE_ODDS_API_KEY='your_api_key'
python -m ufc_predictor.cli fetch-odds --output data/odds_board.csv --regions us --include-links
```

The fetcher uses sport key `mma_mixed_martial_arts`, market `h2h`, and American odds so the output is ready for `rank-odds` or `betting-report`. If you prefer specific books, use `--bookmakers draftkings,fanduel` instead of `--regions us`.

## Raw Table Schema

The core pipeline reads these CSVs from `data/raw/`:

- `events.csv`: event metadata and dates
- `fighters.csv`: fighter profile data such as DOB, height, reach, stance
- `fights.csv`: matchup, result, method, weight class, scheduled rounds
- `fight_stats.csv`: per-fighter fight totals plus optional late-round aggregates

The `download-ufcstats` command converts the public mirror into this schema.

## Tests

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -v
python -m compileall -q src tests
```

## Documentation

- [Model Card](docs/MODEL_CARD.md)
- [Data Notes](docs/DATA.md)
- [Results](docs/RESULTS.md)
- [Odds And Risk](docs/ODDS.md)

## Caveats

UFC prediction is noisy. Public models often report inflated results because they accidentally include future fight information in historical averages. This project is designed around rolling, pre-fight features, but it still lacks private/contextual signals such as injuries, camp quality, short-notice changes, current betting market movement, and medical information.

## License

MIT. See [LICENSE](LICENSE).
