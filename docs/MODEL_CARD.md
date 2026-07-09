# Model Card

## Intended Use

UFC Predictor estimates win probabilities for scheduled UFC matchups using historical UFCStats-style data. It is intended for sports analytics, experimentation, and model-development learning.

It is not intended to provide financial, betting, medical, or professional advisory decisions.

## Model

The current primary model is `order_balanced_stamina_logistic`:

- Logistic regression with numeric scaling and categorical one-hot encoding
- Strong regularization selected after adding stamina/fade features
- Order-balanced training by mirroring fighter A/B matchup orientation
- Final saved model is fit on all trainable rows after reporting chronological holdout metrics
- A separate odds-ranking layer compares model probabilities with sportsbook lines for expected-value analytics

## Features

The model uses rolling pre-fight features only:

- Fighter demographics: age, height, reach, weight, stance
- Experience and activity: UFC fights, losses, layoff days, UFC tenure
- Recent form: recent win rate and win/loss streaks
- Strength of schedule: average prior opponent Elo
- Striking: landed, absorbed, accuracy, defense, differentials
- Grappling: takedowns, accuracy, defense, absorbed takedowns
- Control: control time for/against and differentials
- Stamina/fade proxies: round 3+ and round 4-5 strike, takedown, and control performance
- Optional context hooks: short notice, weight misses, camp changes, disclosed injuries, camps, altitude, and travel distance when supplied as pre-fight data

## Evaluation

The main benchmark uses a chronological holdout: the latest 20% of trainable historical fights are held out before final refit.

Latest local benchmark:

- Accuracy: 63.55%
- Log loss: 0.6441
- ROC AUC: 0.6695
- Holdout rows: 1,709

Additional evaluation tools include model-family comparison, calibration bucket reports, and historical sportsbook-odds bankroll replay when real historical odds are supplied.

## Limitations

The model does not currently include:

- Bundled historical sportsbook closing odds
- Automatic market-movement features
- Automatic injury/camp/weight-cut/short-notice feeds
- Rankings or media scorecards
- News, interviews, or qualitative scouting
- Non-UFC fight history unless provided in compatible raw tables

## Risks

Sports outcomes are inherently noisy. Reported accuracy can vary as new fights are added, source data changes, or validation windows shift. Do not treat output probabilities as guaranteed outcomes.
