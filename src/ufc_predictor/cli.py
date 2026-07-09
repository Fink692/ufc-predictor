from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from .backtest_workbook import generate_backtest_workbook
from .features import FeatureBuilder, build_features_from_raw
from .io import ensure_parent, load_raw_tables, read_csv, write_csv, write_json
from .models import evaluate_model, load_model, save_model, train_model
from .odds import (
    add_no_vig_probabilities,
    build_fight_recommendations,
    fight_recommendations_to_markdown,
    rank_value_bets,
    value_bets_to_markdown,
)
from .odds_api import DEFAULT_MMA_SPORT_KEY, fetch_odds_api_events, odds_api_events_to_board
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

    fetch_odds = subparsers.add_parser("fetch-odds", help="Fetch current MMA moneyline odds from The Odds API")
    fetch_odds.add_argument("--output", default="data/odds_board.csv", help="CSV path for the rankable odds board")
    _add_odds_api_arguments(fetch_odds)

    rank_odds = subparsers.add_parser("rank-odds", help="Rank sportsbook lines by model edge and conservative Kelly sizing")
    rank_odds.add_argument("--predictions", default="reports/predictions.csv")
    rank_odds.add_argument("--odds-board", default="data/odds_board.csv")
    rank_odds.add_argument("--output", default="reports/value_bets.csv")
    rank_odds.add_argument("--bankroll", type=float, default=1000.0)
    rank_odds.add_argument("--kelly-multiplier", type=float, default=0.25)
    rank_odds.add_argument("--max-bankroll-fraction", type=float, default=0.02)
    rank_odds.add_argument("--min-edge", type=float, default=0.02)
    rank_odds.add_argument("--min-expected-roi", type=float, default=0.0)

    betting_report = subparsers.add_parser("betting-report", help="Score fights and generate a ranked odds report")
    betting_report.add_argument("--raw-dir", default="data/raw", help="Directory containing raw UFC history tables")
    betting_report.add_argument("--model-path", default="models/ufc_model.joblib", help="Trained model bundle path")
    betting_report.add_argument("--upcoming", default="data/upcoming_fights.csv", help="Upcoming fights CSV")
    betting_report.add_argument("--odds-board", default="data/odds_board.csv", help="Sportsbook odds board CSV")
    betting_report.add_argument("--fetch-live-odds", action="store_true", help="Fetch live odds into --odds-board before reporting")
    betting_report.add_argument("--predictions-output", default="reports/predictions.csv", help="Prediction CSV output")
    betting_report.add_argument("--output", default="reports/value_bets.csv", help="Ranked value-bet CSV output")
    betting_report.add_argument("--fight-output", default="reports/fight_recommendations.csv", help="Per-fight confidence bet CSV output")
    betting_report.add_argument("--markdown-output", default="reports/betting_report.md", help="Markdown summary output")
    betting_report.add_argument("--top-n", type=int, default=10, help="Number of rows to include in the Markdown summary")
    betting_report.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll used for stake sizing")
    betting_report.add_argument("--max-confidence-stake", type=float, default=100.0, help="Stake at 100%% confidence")
    betting_report.add_argument("--kelly-multiplier", type=float, default=0.25, help="Fraction of full Kelly to use")
    betting_report.add_argument("--max-bankroll-fraction", type=float, default=0.02, help="Maximum stake as bankroll fraction")
    betting_report.add_argument("--min-edge", type=float, default=0.02, help="Minimum model edge over implied probability")
    betting_report.add_argument("--min-expected-roi", type=float, default=0.0, help="Minimum expected return per dollar")
    _add_odds_api_arguments(betting_report)

    workbook = subparsers.add_parser("backtest-workbook", help="Generate an Excel holdout backtest workbook with tables and charts")
    workbook.add_argument("--features", default="data/processed/features.csv", help="Training feature table")
    workbook.add_argument("--output", default="reports/ufc_backtest_tables_charts.xlsx", help="Excel workbook output path")
    workbook.add_argument("--rows-output", default="reports/ufc_holdout_backtest_rows.csv", help="Optional fight-level CSV output")
    workbook.add_argument("--max-confidence-stake", type=float, default=100.0, help="Stake at 100%% confidence")
    workbook.add_argument("--starting-bankroll", type=float, default=1000.0, help="Starting bankroll displayed in the workbook")
    workbook.add_argument("--holdout-fraction", type=float, default=0.2, help="Latest chronological fraction used for holdout")

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
    elif args.command == "fetch-odds":
        run_fetch_odds(
            Path(args.output),
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            sport_key=args.sport_key,
            regions=args.regions,
            bookmakers=args.bookmakers,
            markets=args.markets,
            commence_time_from=args.commence_time_from,
            commence_time_to=args.commence_time_to,
            include_links=args.include_links,
            include_sids=args.include_sids,
            include_bet_limits=args.include_bet_limits,
        )
    elif args.command == "rank-odds":
        run_rank_odds(
            Path(args.predictions),
            Path(args.odds_board),
            Path(args.output),
            bankroll=args.bankroll,
            kelly_multiplier=args.kelly_multiplier,
            max_bankroll_fraction=args.max_bankroll_fraction,
            min_edge=args.min_edge,
            min_expected_roi=args.min_expected_roi,
        )
    elif args.command == "backtest-workbook":
        result = generate_backtest_workbook(
            features_path=Path(args.features),
            output_path=Path(args.output),
            rows_output_path=Path(args.rows_output) if args.rows_output else None,
            max_confidence_stake=args.max_confidence_stake,
            starting_bankroll=args.starting_bankroll,
            holdout_fraction=args.holdout_fraction,
        )
        print(f"Wrote workbook to {result['output_path']}")
        if result["rows_output_path"] is not None:
            print(f"Wrote fight rows to {result['rows_output_path']}")
        print(
            "Holdout summary: "
            f"{result['holdout_rows']} fights, "
            f"accuracy {result['accuracy']:.2%}, "
            f"even-money ROI {result['roi_even_money']:.2%}"
        )
    elif args.command == "betting-report":
        run_betting_report(
            raw_dir=Path(args.raw_dir),
            model_path=Path(args.model_path),
            upcoming_path=Path(args.upcoming),
            odds_board_path=Path(args.odds_board),
            fetch_live_odds=args.fetch_live_odds,
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            sport_key=args.sport_key,
            regions=args.regions,
            bookmakers=args.bookmakers,
            markets=args.markets,
            commence_time_from=args.commence_time_from,
            commence_time_to=args.commence_time_to,
            include_links=args.include_links,
            include_sids=args.include_sids,
            include_bet_limits=args.include_bet_limits,
            predictions_output_path=Path(args.predictions_output),
            output_path=Path(args.output),
            fight_output_path=Path(args.fight_output),
            markdown_output_path=Path(args.markdown_output) if args.markdown_output else None,
            top_n=args.top_n,
            bankroll=args.bankroll,
            max_confidence_stake=args.max_confidence_stake,
            kelly_multiplier=args.kelly_multiplier,
            max_bankroll_fraction=args.max_bankroll_fraction,
            min_edge=args.min_edge,
            min_expected_roi=args.min_expected_roi,
        )


def _add_odds_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", default=None, help="The Odds API key; defaults to --api-key-env")
    parser.add_argument("--api-key-env", default="THE_ODDS_API_KEY", help="Environment variable containing the API key")
    parser.add_argument("--sport-key", default=DEFAULT_MMA_SPORT_KEY, help="The Odds API sport key")
    parser.add_argument("--regions", default="us", help="Bookmaker regions when --bookmakers is not set")
    parser.add_argument("--bookmakers", default=None, help="Comma-separated bookmaker keys, for example draftkings,fanduel")
    parser.add_argument("--markets", default="h2h", help="Comma-separated market keys; h2h is used for fight winner")
    parser.add_argument("--commence-time-from", default=None, help="Optional ISO8601 lower event-time bound")
    parser.add_argument("--commence-time-to", default=None, help="Optional ISO8601 upper event-time bound")
    parser.add_argument("--include-links", action="store_true", help="Request bookmaker and betslip links when available")
    parser.add_argument("--include-sids", action="store_true", help="Request bookmaker source IDs when available")
    parser.add_argument("--include-bet-limits", action="store_true", help="Request bet limits when available")


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
    output = build_prediction_output(raw_dir, model_path, input_path)
    write_csv(output, output_path)
    print(f"Wrote {len(output)} predictions to {output_path}")


def build_prediction_output(raw_dir: Path, model_path: Path, input_path: Path) -> pd.DataFrame:
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
    output["pick_probability"] = output[["prob_fighter_a", "prob_fighter_b"]].max(axis=1)
    output["confidence"] = output["pick_probability"].map(_pick_confidence)
    output["confidence_percent"] = output["confidence"] * 100.0
    output["confidence_bucket"] = output["pick_probability"].map(_confidence_bucket)
    output["model_version"] = str(bundle.metadata.get("model_type", "unknown"))
    return output


def run_fetch_odds(
    output_path: Path,
    api_key: str | None,
    api_key_env: str,
    sport_key: str,
    regions: str,
    bookmakers: str | None,
    markets: str,
    commence_time_from: str | None,
    commence_time_to: str | None,
    include_links: bool,
    include_sids: bool,
    include_bet_limits: bool,
) -> None:
    odds_board = fetch_live_odds_board(
        api_key=api_key,
        api_key_env=api_key_env,
        sport_key=sport_key,
        regions=regions,
        bookmakers=bookmakers,
        markets=markets,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
        include_links=include_links,
        include_sids=include_sids,
        include_bet_limits=include_bet_limits,
    )
    write_csv(odds_board, output_path)
    print(f"Wrote {len(odds_board)} odds rows to {output_path}")


def fetch_live_odds_board(
    api_key: str | None,
    api_key_env: str,
    sport_key: str,
    regions: str,
    bookmakers: str | None,
    markets: str,
    commence_time_from: str | None,
    commence_time_to: str | None,
    include_links: bool,
    include_sids: bool,
    include_bet_limits: bool,
) -> pd.DataFrame:
    resolved_api_key = api_key or os.environ.get(api_key_env, "")
    events = fetch_odds_api_events(
        api_key=resolved_api_key,
        sport_key=sport_key,
        regions=regions,
        bookmakers=bookmakers,
        markets=markets,
        commence_time_from=commence_time_from,
        commence_time_to=commence_time_to,
        include_links=include_links,
        include_sids=include_sids,
        include_bet_limits=include_bet_limits,
    )
    board_market = markets.split(",", maxsplit=1)[0].strip()
    return odds_api_events_to_board(events, market_key=board_market)


def run_rank_odds(
    predictions_path: Path,
    odds_board_path: Path,
    output_path: Path,
    bankroll: float,
    kelly_multiplier: float,
    max_bankroll_fraction: float,
    min_edge: float,
    min_expected_roi: float,
) -> None:
    value_bets = rank_value_bets(
        read_csv(predictions_path),
        read_csv(odds_board_path),
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        max_bankroll_fraction=max_bankroll_fraction,
        min_edge=min_edge,
        min_expected_roi=min_expected_roi,
    )
    write_csv(value_bets, output_path)
    bet_count = int((value_bets["decision"] == "bet").sum()) if not value_bets.empty else 0
    print(f"Wrote {len(value_bets)} ranked odds rows to {output_path} with {bet_count} value candidates")


def run_betting_report(
    raw_dir: Path,
    model_path: Path,
    upcoming_path: Path,
    odds_board_path: Path,
    fetch_live_odds: bool,
    api_key: str | None,
    api_key_env: str,
    sport_key: str,
    regions: str,
    bookmakers: str | None,
    markets: str,
    commence_time_from: str | None,
    commence_time_to: str | None,
    include_links: bool,
    include_sids: bool,
    include_bet_limits: bool,
    predictions_output_path: Path,
    output_path: Path,
    fight_output_path: Path,
    markdown_output_path: Path | None,
    top_n: int,
    bankroll: float,
    max_confidence_stake: float,
    kelly_multiplier: float,
    max_bankroll_fraction: float,
    min_edge: float,
    min_expected_roi: float,
) -> None:
    predictions = build_prediction_output(raw_dir, model_path, upcoming_path)
    write_csv(predictions, predictions_output_path)
    if fetch_live_odds:
        odds_board = fetch_live_odds_board(
            api_key=api_key,
            api_key_env=api_key_env,
            sport_key=sport_key,
            regions=regions,
            bookmakers=bookmakers,
            markets=markets,
            commence_time_from=commence_time_from,
            commence_time_to=commence_time_to,
            include_links=include_links,
            include_sids=include_sids,
            include_bet_limits=include_bet_limits,
        )
        write_csv(odds_board, odds_board_path)
        print(f"Wrote {len(odds_board)} live odds rows to {odds_board_path}")
    else:
        odds_board = read_csv(odds_board_path)
    value_bets = rank_value_bets(
        predictions,
        odds_board,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
        max_bankroll_fraction=max_bankroll_fraction,
        min_edge=min_edge,
        min_expected_roi=min_expected_roi,
    )
    write_csv(value_bets, output_path)
    fight_recommendations = build_fight_recommendations(
        predictions,
        odds_board,
        max_confidence_stake=max_confidence_stake,
    )
    write_csv(fight_recommendations, fight_output_path)
    if markdown_output_path is not None:
        ensure_parent(markdown_output_path)
        markdown_output_path.write_text(
            value_bets_to_markdown(value_bets, top_n=top_n, bankroll=bankroll)
            + "\n"
            + fight_recommendations_to_markdown(fight_recommendations, top_n=top_n),
            encoding="utf-8",
        )
    bet_count = int((value_bets["decision"] == "bet").sum()) if not value_bets.empty else 0
    print(f"Wrote {len(predictions)} predictions to {predictions_output_path}")
    print(f"Wrote {len(value_bets)} ranked odds rows to {output_path} with {bet_count} value candidates")
    print(f"Wrote {len(fight_recommendations)} fight recommendations to {fight_output_path}")
    if markdown_output_path is not None:
        print(f"Wrote betting report to {markdown_output_path}")


def _confidence_bucket(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    if probability >= 0.6:
        return "medium"
    return "low"


def _pick_confidence(probability: float) -> float:
    return max(0.0, min(1.0, (float(probability) - 0.5) * 2.0))


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
