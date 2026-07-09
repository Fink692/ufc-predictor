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

This keeps the repository lightweight and avoids redistributing generated data/model artifacts. Recreate them with the CLI commands in the workflow guide.

## Leakage Policy

Feature generation is event-time ordered. Each matchup row is built from fighter history accumulated before that fight date. The fight being predicted is added to fighter histories only after its feature row is created.

Draws, no contests, and overturned results are excluded from winner-label training.

## Stamina/Fade Data

When source round-level stats are available, conversion preserves:

- Round 3+ aggregates as `late_*`
- Round 4-5 aggregates as `champ_*`

The model uses these as prior-history stamina/fade proxies, not as information from the fight being predicted.

## Optional Context Data

The public UFCStats workflow does not include private/contextual variables, but the feature builder can consume them when reliable pre-fight data is available.

Supported optional columns:

- `fighter_a_short_notice`, `fighter_b_short_notice`
- `fighter_a_weight_miss`, `fighter_b_weight_miss`
- `fighter_a_camp_change`, `fighter_b_camp_change`
- `fighter_a_disclosed_injury`, `fighter_b_disclosed_injury`
- `fighter_a_camp`, `fighter_b_camp`
- `altitude_ft`
- `fighter_a_travel_distance_km`, `fighter_b_travel_distance_km`, or `travel_distance_diff_km`

Missing optional values default to neutral values. These columns should only contain information known before the fight.

## Historical Odds Data

Historical odds are not bundled because reliable closing-line data is usually licensed or API-provided. The project supports importing a historical odds board with:

- `event_date`
- `fighter_a`
- `fighter_b`
- `sportsbook`
- `fighter`
- `american_odds`

For a true betting backtest, pair that odds board with historical predictions that include either `actual_winner` or `target_fighter_a_win`.
