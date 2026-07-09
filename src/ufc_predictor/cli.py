from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import FeatureBuilder, build_features_from_raw
from .io import load_raw_tables, read_csv, write_csv, write_json
from .models import evaluate_model, load_model, save_model, train_model
from .odds import add_no_vig_probabilities
from .evaluation import classification_metrics
from .ufcstats_source import DEFAULT_MIRROR_BASE_URL, convert_ufcstats_mirror


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ufc-predict", description="UFC winner prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Validate and copy raw source tables into a processed area")
    ingest.add_argument("--raw-dir", default="data/raw")
    ingest.add_argument("--processed-dir", default="data/processed")

    download = subparsers.add_parser("download-ufcstats", help="Convert a public UFCStats CSV mirror into raw tables")
    download.add_argument("--output-raw-dir", default="data/raw")
    download.add_argument("--base-url", default=DEFAULT_MIRROR_BASE_URL)

    build = subparsers.add_parser("build-features", help="Build leakage-safe training features")
    build.add_argument("--raw-dir", default="data/raw")
    build.add_argument("--output", default="data/processed/features.csv")

    train = subparsers.add_parser("train", help="Train primary and baseline models")
    train.add_argument("--features", default="data/processed/features.csv")
    train.add_argument("--model-path", default="models/ufc_model.joblib")
    train.add_argument("--report", default="reports/train_metrics.json")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model against a feature table")
    evaluate.add_argument("--features", default="data/processed/features.csv")
    evaluate.add_argument("--model-path", default="models/ufc_model.joblib")
    evaluate.add_argument("--report", default="reports/evaluation.json")
    evaluate.add_argument("--odds", default=None, help="Optional odds CSV with fight_id, odds_fighter_a, odds_fighter_b")

    predict = subparsers.add_parser("predict", help="Score upcoming fights from a CSV")
    predict.add_argument("--raw-dir", default="data/raw")
    predict.add_argument("--model-path", default="models/ufc_model.joblib")
    predict.add_argument("--input", default="data/upcoming_fights.csv")
    predict.add_argument("--output", default="reports/predictions.csv")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        run_ingest(Path(args.raw_dir), Path(args.processed_dir))
    elif args.command == "download-ufcstats":
        run_download_ufcstats(Path(args.output_raw_dir), args.base_url)
    elif args.command == "build-features":
        run_build_features(Path(args.raw_dir), Path(args.output))
    elif args.command == "train":
        run_train(Path(args.features), Path(args.model_path), Path(args.report))
    elif args.command == "evaluate":
        run_evaluate(Path(args.features), Path(args.model_path), Path(args.report), Path(args.odds) if args.odds else None)
    elif args.command == "predict":
        run_predict(Path(args.raw_dir), Path(args.model_path), Path(args.input), Path(args.output))


def run_ingest(raw_dir: Path, processed_dir: Path) -> None:
    tables = load_raw_tables(raw_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        write_csv(table, processed_dir / f"{name}.csv")
    print(f"Validated and copied {len(tables)} raw tables to {processed_dir}")


def run_download_ufcstats(output_raw_dir: Path, base_url: str) -> None:
    metadata = convert_ufcstats_mirror(output_raw_dir, base_url=base_url)
    print(
        "Wrote UFCStats raw tables to "
        f"{output_raw_dir} from {metadata['fights']} fights and {metadata['fighters']} fighters"
    )


def run_build_features(raw_dir: Path, output: Path) -> None:
    features = build_features_from_raw(raw_dir)
    write_csv(features, output)
    print(f"Wrote {len(features)} feature rows to {output}")


def run_train(features_path: Path, model_path: Path, report_path: Path) -> None:
    features = read_csv(features_path)
    bundle, metrics = train_model(features)
    save_model(bundle, model_path)
    write_json({"metadata": bundle.metadata, "metrics": metrics}, report_path)
    print(f"Saved model to {model_path}")
    print(f"Wrote training report to {report_path}")


def run_evaluate(features_path: Path, model_path: Path, report_path: Path, odds_path: Path | None = None) -> None:
    features = read_csv(features_path)
    bundle = load_model(model_path)
    report = evaluate_model(bundle, features)
    if odds_path is not None:
        report["market_odds_baseline"] = _market_odds_baseline(features, read_csv(odds_path))
    write_json(report, report_path)
    print(f"Wrote evaluation report to {report_path}")


def run_predict(raw_dir: Path, model_path: Path, input_path: Path, output_path: Path) -> None:
    tables = load_raw_tables(raw_dir)
    upcoming = pd.read_csv(input_path)
    prediction_features = FeatureBuilder().build_prediction_frame(tables, upcoming)
    bundle = load_model(model_path)
    probabilities = bundle.predict_proba(prediction_features)
    output = prediction_features[
        [
            "event_date",
            "fighter_a",
            "fighter_b",
            "weight_class",
            "gender",
            "scheduled_rounds",
            "title_fight",
            "data_quality_flags",
        ]
    ].copy()
    output["prob_fighter_a"] = probabilities
    output["prob_fighter_b"] = 1.0 - probabilities
    output["pick"] = output.apply(
        lambda row: row["fighter_a"] if row["prob_fighter_a"] >= row["prob_fighter_b"] else row["fighter_b"],
        axis=1,
    )
    output["confidence_bucket"] = output[["prob_fighter_a", "prob_fighter_b"]].max(axis=1).map(_confidence_bucket)
    output["model_version"] = str(bundle.metadata.get("model_type", "unknown"))
    write_csv(output, output_path)
    print(f"Wrote {len(output)} predictions to {output_path}")


def _confidence_bucket(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.6:
        return "medium"
    return "low"


def _market_odds_baseline(features: pd.DataFrame, odds: pd.DataFrame) -> dict[str, object]:
    required = {"fight_id", "odds_fighter_a", "odds_fighter_b"}
    missing = sorted(required - set(odds.columns))
    if missing:
        raise ValueError(f"Odds CSV missing columns: {missing}")
    enriched = add_no_vig_probabilities(odds)
    joined = features[["fight_id", "target_fighter_a_win"]].merge(
        enriched[["fight_id", "market_prob_fighter_a", "market_prob_fighter_b"]],
        on="fight_id",
        how="inner",
    )
    if joined.empty:
        return {"metrics": None, "coverage_rows": 0, "note": "No odds rows matched feature fight_id values."}
    return {
        "metrics": classification_metrics(joined["target_fighter_a_win"], joined["market_prob_fighter_a"]),
        "coverage_rows": int(len(joined)),
        "mean_market_prob_fighter_a": float(joined["market_prob_fighter_a"].mean()),
    }


if __name__ == "__main__":
    main()
