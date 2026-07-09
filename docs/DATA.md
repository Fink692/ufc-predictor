# Data Notes

## Source

The built-in public-data command reads CSVs from:

- [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats)

That repository mirrors UFCStats-derived tables. This project converts those files into a stable local schema under `data/raw/`.

## Generated Files

The following are generated and intentionally ignored by git:

- `data/raw/`
- `data/processed/`
- `models/`
- `reports/`

This keeps the repository lightweight and avoids redistributing generated data/model artifacts. Recreate them with the CLI commands in the README.

## Leakage Policy

Feature generation is event-time ordered. Each matchup row is built from fighter history accumulated before that fight date. The fight being predicted is added to fighter histories only after its feature row is created.

Draws, no contests, and overturned results are excluded from winner-label training.

## Stamina/Fade Data

When source round-level stats are available, conversion preserves:

- Round 3+ aggregates as `late_*`
- Round 4-5 aggregates as `champ_*`

The model uses these as prior-history stamina/fade proxies, not as information from the fight being predicted.
