from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .calibration import calibration_summary
from .evaluation import classification_metrics
from .features import MODEL_FEATURES
from .models import build_preprocessor, make_order_balanced_frame


def compare_model_families(
    features: pd.DataFrame,
    holdout_fraction: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    if "target_fighter_a_win" not in features:
        raise ValueError("features must contain target_fighter_a_win.")

    ordered = features.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    test_size = max(1, int(round(len(ordered) * holdout_fraction)))
    train = ordered.iloc[:-test_size].copy()
    holdout = ordered.iloc[-test_size:].copy()
    if len(train) < 2:
        raise ValueError("Not enough rows before the holdout split.")
    y_train = train["target_fighter_a_win"].astype(int)
    if len(set(y_train.tolist())) < 2 or int(y_train.value_counts().min()) < 2:
        raise ValueError("Training split must contain at least two examples of both classes.")

    balanced_train = make_order_balanced_frame(train)
    y_holdout = holdout["target_fighter_a_win"].astype(int)
    rows: list[dict[str, object]] = []
    for name, builder in _model_builders(random_state).items():
        model = builder()
        model.fit(balanced_train[MODEL_FEATURES], balanced_train["target_fighter_a_win"].astype(int))
        probabilities = model.predict_proba(holdout[MODEL_FEATURES])[:, 1]
        metrics = classification_metrics(y_holdout, probabilities)
        calibration = calibration_summary(y_holdout, probabilities, bins=10)
        rows.append(
            {
                "model": name,
                "train_rows": int(len(train)),
                "fit_rows_order_balanced": int(len(balanced_train)),
                "holdout_rows": int(len(holdout)),
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "roc_auc": metrics["roc_auc"],
                "expected_calibration_error": calibration["expected_calibration_error"],
                "maximum_calibration_error": calibration["maximum_calibration_error"],
            }
        )
    results = pd.DataFrame(rows)
    return results.sort_values(["log_loss", "brier_score", "accuracy"], ascending=[True, True, False]).reset_index(drop=True)


def _model_builders(random_state: int) -> dict[str, Callable[[], Pipeline]]:
    return {
        "regularized_logistic": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=4000, C=0.02)),
            ]
        ),
        "balanced_logistic": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("classifier", LogisticRegression(max_iter=4000, C=1.0)),
            ]
        ),
        "random_forest": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        min_samples_leaf=8,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": lambda: Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=False)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.05,
                        l2_regularization=0.2,
                        min_samples_leaf=12,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
