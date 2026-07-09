# Backtest Workbook

The project can generate an Excel workbook with holdout results, tables, and charts:

```powershell
python -m ufc_predictor.cli backtest-workbook --features data/processed/features.csv --output reports/ufc_backtest_tables_charts.xlsx --rows-output reports/ufc_holdout_backtest_rows.csv
```

The published example workbook is available at:

- [Backtest workbook](artifacts/ufc_backtest_tables_charts.xlsx)
- [Fight-level rows CSV](artifacts/ufc_holdout_backtest_rows.csv)

## What It Shows

- Chronological holdout performance using the latest 20% of the feature table
- One row per holdout fight
- The model pick, actual winner, pick probability, adjusted confidence, and model-implied fair odds
- Confidence stake sizing where 100% confidence maps to `$100` and 0% maps to `$0`
- Synthetic even-money profit/loss if every confidence-sized pick were paid at `+100`
- Summary, confidence bucket, yearly, threshold, and chart sheets

## Important Caveat

This is not a historical sportsbook closing-line backtest. The repository does not include a full historical sportsbook odds file. The workbook shows model-implied fair odds and an even-money confidence-stake simulation so the model behavior can be inspected consistently.

For a real betting backtest, add historical odds with fight IDs, sportsbook, closing price, and timestamp, then compare model probabilities against those actual prices.
