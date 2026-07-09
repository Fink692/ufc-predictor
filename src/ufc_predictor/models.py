from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import classification_metrics
from .features import CATEGORICAL_MODEL_FEATURES, MODEL_FEATURES, NUMERIC_MODEL_FEATURES
from .io import ensure_parent


ALWAYS_SIGNED_FEATURES = {
    "age_diff",
    "height_diff_cm",
    "reach_diff_cm",
    "weight_diff_lbs",
    "elo_diff",
}


@dataclass
class ModelBundle:
    model: Pipeline
    logistic_baseline: Pipeline
    feature_columns: list[str]
    metadata: dict[str, object]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features[self.feature_columns])[:, 1]


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_MODEL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_MODEL_FEATURES),
        ],
        remainder="drop",
    )


def train_model(features: pd.DataFrame, random_state: int = 42) -> tuple[ModelBundle, dict[str, object]]:
    if "target_fighter_a_win" not in features.columns:
        raise ValueError("Training features must contain target_fighter_a_win.")
    training = features.sort_values("event_date").reset_index(drop=True)
    X = training[MODEL_FEATURES]
    y = training["target_fighter_a_win"].astype(int)
    if len(set(y.tolist())) < 2:
        raise ValueError("Training data must contain both winner classes.")

    test_size = max(1, int(round(len(training) * 0.2)))
    if (
        len(training) - test_size < 2
        or len(set(y.iloc[:-test_size].tolist())) < 2
        or int(y.iloc[:-test_size].value_counts().min()) < 2
    ):
        test_size = 0

    logistic = _build_logistic_pipeline()
    primary = _build_primary_pipeline(random_state=random_state)

    if test_size:
        train_idx = training.index[:-test_size]
        holdout_idx = training.index[-test_size:]
        balanced_train = make_order_balanced_frame(training.loc[train_idx])
        primary.fit(balanced_train[MODEL_FEATURES], balanced_train["target_fighter_a_win"].astype(int))
        logistic.fit(balanced_train[MODEL_FEATURES], balanced_train["target_fighter_a_win"].astype(int))
        metrics = {
            "holdout": classification_metrics(y.loc[holdout_idx], primary.predict_proba(X.loc[holdout_idx])[:, 1]),
            "logistic_baseline_holdout": classification_metrics(y.loc[holdout_idx], logistic.predict_proba(X.loc[holdout_idx])[:, 1]),
            "fifty_fifty_holdout": classification_metrics(y.loc[holdout_idx], np.repeat(0.5, len(holdout_idx))),
            "elo_holdout": classification_metrics(y.loc[holdout_idx], training.loc[holdout_idx, "elo_prob_fighter_a"]),
        }
    else:
        metrics = {"holdout": None, "note": "Dataset too small for a time-based holdout split."}

    balanced_all = make_order_balanced_frame(training)
    primary.fit(balanced_all[MODEL_FEATURES], balanced_all["target_fighter_a_win"].astype(int))
    logistic.fit(balanced_all[MODEL_FEATURES], balanced_all["target_fighter_a_win"].astype(int))
    metadata = {
        "model_type": "order_balanced_stamina_logistic",
        "row_count": int(len(training)),
        "fit_row_count": int(len(balanced_all)),
        "feature_columns": MODEL_FEATURES,
        "target": "target_fighter_a_win",
        "order_balanced_training": True,
        "random_state": random_state,
    }
    return ModelBundle(primary, logistic, MODEL_FEATURES.copy(), metadata), metrics


def evaluate_model(bundle: ModelBundle, features: pd.DataFrame) -> dict[str, object]:
    evaluation = features.sort_values("event_date").reset_index(drop=True)
    y = evaluation["target_fighter_a_win"].astype(int)
    probabilities = bundle.predict_proba(evaluation)
    return {
        "model": classification_metrics(y, probabilities),
        "logistic_baseline": classification_metrics(
            y,
            bundle.logistic_baseline.predict_proba(evaluation[bundle.feature_columns])[:, 1],
        ),
        "fifty_fifty_baseline": classification_metrics(y, np.repeat(0.5, len(evaluation))),
        "elo_baseline": classification_metrics(y, evaluation["elo_prob_fighter_a"]),
        "walk_forward_event_time": walk_forward_evaluate(evaluation),
    }


def walk_forward_evaluate(
    features: pd.DataFrame,
    min_train_rows: int = 8,
    random_state: int = 42,
) -> dict[str, object]:
    ordered = features.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    predictions: list[float] = []
    targets: list[int] = []
    event_dates: list[str] = []
    folds: list[dict[str, object]] = []

    for event_date in ordered["event_date"].drop_duplicates().tolist():
        train = ordered[ordered["event_date"] < event_date]
        holdout = ordered[ordered["event_date"] == event_date]
        if len(train) < min_train_rows:
            continue
        y_train = train["target_fighter_a_win"].astype(int)
        if len(set(y_train.tolist())) < 2 or int(y_train.value_counts().min()) < 2:
            continue
        balanced_train = make_order_balanced_frame(train)
        model = _build_primary_pipeline(random_state=random_state)
        model.fit(balanced_train[MODEL_FEATURES], balanced_train["target_fighter_a_win"].astype(int))
        fold_probabilities = model.predict_proba(holdout[MODEL_FEATURES])[:, 1]
        predictions.extend(fold_probabilities.tolist())
        targets.extend(holdout["target_fighter_a_win"].astype(int).tolist())
        event_dates.extend([str(event_date)] * len(holdout))
        folds.append(
            {
                "event_date": str(event_date),
                "train_rows": int(len(train)),
                "holdout_rows": int(len(holdout)),
            }
        )

    if not predictions:
        return {
            "metrics": None,
            "folds": [],
            "note": "Not enough prior event-time data with both classes for walk-forward evaluation.",
        }
    return {
        "metrics": classification_metrics(targets, predictions),
        "folds": folds,
        "prediction_rows": [
            {"event_date": event_date, "target": int(target), "prob_fighter_a": float(probability)}
            for event_date, target, probability in zip(event_dates, targets, predictions)
        ],
    }


def save_model(bundle: ModelBundle, path: str | Path) -> None:
    output = Path(path)
    ensure_parent(output)
    joblib.dump(bundle, output)


def load_model(path: str | Path) -> ModelBundle:
    return joblib.load(Path(path))


def make_order_balanced_frame(features: pd.DataFrame) -> pd.DataFrame:
    swapped = features.copy()
    for left in [column for column in features.columns if column.startswith("fighter_a_")]:
        right = "fighter_b_" + left.removeprefix("fighter_a_")
        if right in features.columns:
            swapped[left] = features[right].to_numpy()
            swapped[right] = features[left].to_numpy()
    for left, right in [
        ("fighter_a", "fighter_b"),
    ]:
        if left in features.columns and right in features.columns:
            swapped[left] = features[right].to_numpy()
            swapped[right] = features[left].to_numpy()
    for column in features.columns:
        if column in ALWAYS_SIGNED_FEATURES or column.endswith("_diff") or column.endswith("_diff_days"):
            swapped[column] = -features[column]
    if "elo_prob_fighter_a" in features.columns:
        swapped["elo_prob_fighter_a"] = 1.0 - features["elo_prob_fighter_a"]
    if "target_fighter_a_win" in features.columns:
        swapped["target_fighter_a_win"] = 1 - features["target_fighter_a_win"].astype(int)
    return pd.concat([features, swapped], ignore_index=True)


def _build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=4000, C=1.0)),
        ]
    )


def _build_primary_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=4000, C=0.02)),
        ]
    )
