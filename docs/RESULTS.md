# Results

## Current Benchmark

Latest local run on the public UFCStats mirror:

| Item | Value |
| --- | ---: |
| Source fights | 8,701 |
| Trainable fights | 8,547 |
| Date range | 1994-03-11 to 2026-05-16 |
| Holdout split | latest chronological 20% |
| Holdout rows | 1,709 |
| Model features | 148 active in benchmark; 165 current schema columns with optional context hooks |

| Metric | Model | 50/50 Baseline | Elo Baseline |
| --- | ---: | ---: | ---: |
| Accuracy | 63.55% | 55.71% | 55.30% |
| Log loss | 0.6441 | 0.6931 | 0.6841 |
| ROC AUC | 0.6695 | 0.5000 | 0.5590 |

## Recent Improvements

| Version | Accuracy | Log Loss | ROC AUC |
| --- | ---: | ---: | ---: |
| Initial large model | 61.73% | 0.6614 | 0.6393 |
| Richer history + ensemble | 63.37% | 0.6564 | 0.6451 |
| Stamina/fade features + regularized logistic | 63.55% | 0.6441 | 0.6695 |

The stamina/fade features gave only a small raw accuracy lift, but materially improved probability ranking and calibration metrics.

## Added Evaluation Tooling

The project now includes:

- Model-family comparison across regularized logistic regression, logistic baseline, random forest, and histogram gradient boosting
- Calibration bucket reports with expected calibration error and maximum calibration error
- Historical odds replay for real moneyline data with conservative Kelly, flat-stake, and confidence-stake bankroll simulations

The published benchmark above remains the last large public-data run. Optional context hooks default to neutral values unless reliable pre-fight context data is supplied.
