from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

from .elo import EloTracker
from .io import load_raw_tables, normalize_name


OUTCOME_EXCLUDE = {"draw", "no contest", "nc", "overturned"}

NUMERIC_MODEL_FEATURES = [
    "age_diff",
    "height_diff_cm",
    "reach_diff_cm",
    "weight_diff_lbs",
    "fighter_a_ufc_fights",
    "fighter_b_ufc_fights",
    "experience_diff",
    "fighter_a_ufc_losses",
    "fighter_b_ufc_losses",
    "loss_diff",
    "fighter_a_win_rate",
    "fighter_b_win_rate",
    "win_rate_diff",
    "fighter_a_recent_win_rate_3",
    "fighter_b_recent_win_rate_3",
    "recent_win_rate_3_diff",
    "fighter_a_recent_win_rate_5",
    "fighter_b_recent_win_rate_5",
    "recent_win_rate_5_diff",
    "fighter_a_current_win_streak",
    "fighter_b_current_win_streak",
    "current_win_streak_diff",
    "fighter_a_current_loss_streak",
    "fighter_b_current_loss_streak",
    "current_loss_streak_diff",
    "fighter_a_finish_rate",
    "fighter_b_finish_rate",
    "finish_rate_diff",
    "fighter_a_layoff_days",
    "fighter_b_layoff_days",
    "layoff_diff_days",
    "fighter_a_days_since_ufc_debut",
    "fighter_b_days_since_ufc_debut",
    "days_since_ufc_debut_diff",
    "fighter_a_avg_fight_minutes",
    "fighter_b_avg_fight_minutes",
    "avg_fight_minutes_diff",
    "fighter_a_late_round_minutes",
    "fighter_b_late_round_minutes",
    "late_round_minutes_diff",
    "fighter_a_late_round_share",
    "fighter_b_late_round_share",
    "late_round_share_diff",
    "fighter_a_champ_round_minutes",
    "fighter_b_champ_round_minutes",
    "champ_round_minutes_diff",
    "fighter_a_avg_opponent_elo",
    "fighter_b_avg_opponent_elo",
    "avg_opponent_elo_diff",
    "fighter_a_strikes_landed_per_15",
    "fighter_b_strikes_landed_per_15",
    "strikes_landed_per_15_diff",
    "fighter_a_strikes_absorbed_per_15",
    "fighter_b_strikes_absorbed_per_15",
    "strikes_absorbed_per_15_diff",
    "fighter_a_strike_differential_per_15",
    "fighter_b_strike_differential_per_15",
    "strike_differential_per_15_diff",
    "fighter_a_strike_accuracy",
    "fighter_b_strike_accuracy",
    "strike_accuracy_diff",
    "fighter_a_strike_defense",
    "fighter_b_strike_defense",
    "strike_defense_diff",
    "fighter_a_total_strikes_landed_per_15",
    "fighter_b_total_strikes_landed_per_15",
    "total_strikes_landed_per_15_diff",
    "fighter_a_total_strikes_absorbed_per_15",
    "fighter_b_total_strikes_absorbed_per_15",
    "total_strikes_absorbed_per_15_diff",
    "fighter_a_total_strike_differential_per_15",
    "fighter_b_total_strike_differential_per_15",
    "total_strike_differential_per_15_diff",
    "fighter_a_knockdowns_for_per_15",
    "fighter_b_knockdowns_for_per_15",
    "knockdowns_for_per_15_diff",
    "fighter_a_knockdowns_against_per_15",
    "fighter_b_knockdowns_against_per_15",
    "knockdowns_against_per_15_diff",
    "fighter_a_knockdown_differential_per_15",
    "fighter_b_knockdown_differential_per_15",
    "knockdown_differential_per_15_diff",
    "fighter_a_takedowns_per_15",
    "fighter_b_takedowns_per_15",
    "takedowns_per_15_diff",
    "fighter_a_takedown_accuracy",
    "fighter_b_takedown_accuracy",
    "takedown_accuracy_diff",
    "fighter_a_takedowns_absorbed_per_15",
    "fighter_b_takedowns_absorbed_per_15",
    "takedowns_absorbed_per_15_diff",
    "fighter_a_takedown_defense",
    "fighter_b_takedown_defense",
    "takedown_defense_diff",
    "fighter_a_takedown_differential_per_15",
    "fighter_b_takedown_differential_per_15",
    "takedown_differential_per_15_diff",
    "fighter_a_submission_attempts_per_15",
    "fighter_b_submission_attempts_per_15",
    "submission_attempts_per_15_diff",
    "fighter_a_control_seconds_per_15",
    "fighter_b_control_seconds_per_15",
    "control_seconds_per_15_diff",
    "fighter_a_control_absorbed_seconds_per_15",
    "fighter_b_control_absorbed_seconds_per_15",
    "control_absorbed_seconds_per_15_diff",
    "fighter_a_control_differential_seconds_per_15",
    "fighter_b_control_differential_seconds_per_15",
    "control_differential_seconds_per_15_diff",
    "fighter_a_late_strikes_landed_per_5",
    "fighter_b_late_strikes_landed_per_5",
    "late_strikes_landed_per_5_diff",
    "fighter_a_late_strikes_absorbed_per_5",
    "fighter_b_late_strikes_absorbed_per_5",
    "late_strikes_absorbed_per_5_diff",
    "fighter_a_late_strike_differential_per_5",
    "fighter_b_late_strike_differential_per_5",
    "late_strike_differential_per_5_diff",
    "fighter_a_late_strike_accuracy",
    "fighter_b_late_strike_accuracy",
    "late_strike_accuracy_diff",
    "fighter_a_late_strike_defense",
    "fighter_b_late_strike_defense",
    "late_strike_defense_diff",
    "fighter_a_late_takedowns_per_5",
    "fighter_b_late_takedowns_per_5",
    "late_takedowns_per_5_diff",
    "fighter_a_late_takedown_defense",
    "fighter_b_late_takedown_defense",
    "late_takedown_defense_diff",
    "fighter_a_late_control_differential_seconds_per_5",
    "fighter_b_late_control_differential_seconds_per_5",
    "late_control_differential_seconds_per_5_diff",
    "fighter_a_champ_strike_differential_per_5",
    "fighter_b_champ_strike_differential_per_5",
    "champ_strike_differential_per_5_diff",
    "fighter_a_champ_control_differential_seconds_per_5",
    "fighter_b_champ_control_differential_seconds_per_5",
    "champ_control_differential_seconds_per_5_diff",
    "elo_prob_fighter_a",
    "elo_diff",
    "scheduled_rounds",
    "title_fight",
    "same_stance",
]

CATEGORICAL_MODEL_FEATURES = [
    "weight_class",
    "gender",
    "fighter_a_stance",
    "fighter_b_stance",
]

MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES


@dataclass
class FighterHistory:
    fights: int = 0
    wins: int = 0
    losses: int = 0
    finishes: int = 0
    result_history: list[int] = field(default_factory=list)
    strikes_att: float = 0.0
    strikes_succ: float = 0.0
    strikes_att_against: float = 0.0
    strikes_succ_against: float = 0.0
    total_strikes_att: float = 0.0
    total_strikes_succ: float = 0.0
    total_strikes_att_against: float = 0.0
    total_strikes_succ_against: float = 0.0
    knockdowns_for: float = 0.0
    knockdowns_against: float = 0.0
    takedown_att: float = 0.0
    takedown_succ: float = 0.0
    takedown_att_against: float = 0.0
    takedown_succ_against: float = 0.0
    submission_att: float = 0.0
    reversals: float = 0.0
    ctrl_time_seconds: float = 0.0
    ctrl_time_seconds_against: float = 0.0
    fight_minutes: float = 0.0
    late_fight_minutes: float = 0.0
    late_strikes_att: float = 0.0
    late_strikes_succ: float = 0.0
    late_strikes_att_against: float = 0.0
    late_strikes_succ_against: float = 0.0
    late_takedown_att: float = 0.0
    late_takedown_succ: float = 0.0
    late_takedown_att_against: float = 0.0
    late_takedown_succ_against: float = 0.0
    late_ctrl_time_seconds: float = 0.0
    late_ctrl_time_seconds_against: float = 0.0
    champ_fight_minutes: float = 0.0
    champ_strikes_succ: float = 0.0
    champ_strikes_succ_against: float = 0.0
    champ_ctrl_time_seconds: float = 0.0
    champ_ctrl_time_seconds_against: float = 0.0
    opponent_elo_sum: float = 0.0
    first_fight_date: pd.Timestamp | None = None
    last_fight_date: pd.Timestamp | None = None

    def snapshot(self, event_date: pd.Timestamp) -> dict[str, float]:
        fight_minutes = max(self.fight_minutes, 1.0)
        layoff = np.nan
        if self.last_fight_date is not None:
            layoff = float((event_date - self.last_fight_date).days)
        days_since_debut = np.nan
        if self.first_fight_date is not None:
            days_since_debut = float((event_date - self.first_fight_date).days)
        return {
            "ufc_fights": float(self.fights),
            "ufc_losses": float(self.losses),
            "win_rate": self.wins / self.fights if self.fights else 0.5,
            "recent_win_rate_3": _recent_win_rate(self.result_history, 3),
            "recent_win_rate_5": _recent_win_rate(self.result_history, 5),
            "current_win_streak": float(_current_streak(self.result_history, 1)),
            "current_loss_streak": float(_current_streak(self.result_history, 0)),
            "finish_rate": self.finishes / self.wins if self.wins else 0.0,
            "layoff_days": layoff,
            "days_since_ufc_debut": days_since_debut,
            "avg_fight_minutes": self.fight_minutes / self.fights if self.fights else 0.0,
            "late_round_minutes": self.late_fight_minutes,
            "late_round_share": self.late_fight_minutes / self.fight_minutes if self.fight_minutes else 0.0,
            "champ_round_minutes": self.champ_fight_minutes,
            "avg_opponent_elo": self.opponent_elo_sum / self.fights if self.fights else 1500.0,
            "knockdowns_for_per_15": self.knockdowns_for / fight_minutes * 15.0,
            "knockdowns_against_per_15": self.knockdowns_against / fight_minutes * 15.0,
            "knockdown_differential_per_15": (self.knockdowns_for - self.knockdowns_against) / fight_minutes * 15.0,
            "strikes_landed_per_15": self.strikes_succ / fight_minutes * 15.0,
            "strikes_absorbed_per_15": self.strikes_succ_against / fight_minutes * 15.0,
            "strike_differential_per_15": (self.strikes_succ - self.strikes_succ_against) / fight_minutes * 15.0,
            "strike_accuracy": self.strikes_succ / self.strikes_att if self.strikes_att else 0.0,
            "strike_defense": 1.0 - (self.strikes_succ_against / self.strikes_att_against) if self.strikes_att_against else 0.0,
            "total_strikes_landed_per_15": self.total_strikes_succ / fight_minutes * 15.0,
            "total_strikes_absorbed_per_15": self.total_strikes_succ_against / fight_minutes * 15.0,
            "total_strike_differential_per_15": (self.total_strikes_succ - self.total_strikes_succ_against) / fight_minutes * 15.0,
            "takedowns_per_15": self.takedown_succ / fight_minutes * 15.0,
            "takedown_accuracy": self.takedown_succ / self.takedown_att if self.takedown_att else 0.0,
            "takedowns_absorbed_per_15": self.takedown_succ_against / fight_minutes * 15.0,
            "takedown_defense": 1.0 - (self.takedown_succ_against / self.takedown_att_against) if self.takedown_att_against else 0.0,
            "takedown_differential_per_15": (self.takedown_succ - self.takedown_succ_against) / fight_minutes * 15.0,
            "submission_attempts_per_15": self.submission_att / fight_minutes * 15.0,
            "control_seconds_per_15": self.ctrl_time_seconds / fight_minutes * 15.0,
            "control_absorbed_seconds_per_15": self.ctrl_time_seconds_against / fight_minutes * 15.0,
            "control_differential_seconds_per_15": (self.ctrl_time_seconds - self.ctrl_time_seconds_against) / fight_minutes * 15.0,
            "late_strikes_landed_per_5": _rate_per(self.late_strikes_succ, self.late_fight_minutes),
            "late_strikes_absorbed_per_5": _rate_per(self.late_strikes_succ_against, self.late_fight_minutes),
            "late_strike_differential_per_5": _rate_per(self.late_strikes_succ - self.late_strikes_succ_against, self.late_fight_minutes),
            "late_strike_accuracy": self.late_strikes_succ / self.late_strikes_att if self.late_strikes_att else 0.0,
            "late_strike_defense": 1.0 - (self.late_strikes_succ_against / self.late_strikes_att_against) if self.late_strikes_att_against else 0.0,
            "late_takedowns_per_5": _rate_per(self.late_takedown_succ, self.late_fight_minutes),
            "late_takedown_defense": 1.0 - (self.late_takedown_succ_against / self.late_takedown_att_against) if self.late_takedown_att_against else 0.0,
            "late_control_differential_seconds_per_5": _rate_per(
                self.late_ctrl_time_seconds - self.late_ctrl_time_seconds_against,
                self.late_fight_minutes,
            ),
            "champ_strike_differential_per_5": _rate_per(
                self.champ_strikes_succ - self.champ_strikes_succ_against,
                self.champ_fight_minutes,
            ),
            "champ_control_differential_seconds_per_5": _rate_per(
                self.champ_ctrl_time_seconds - self.champ_ctrl_time_seconds_against,
                self.champ_fight_minutes,
            ),
        }

    def update(
        self,
        won: bool,
        finish_win: bool,
        stats: dict[str, float],
        opponent_stats: dict[str, float],
        event_date: pd.Timestamp,
        opponent_elo: float | None = None,
        official_decision: bool = True,
    ) -> None:
        self.fights += 1
        self.wins += int(won and official_decision)
        self.losses += int((not won) and official_decision)
        self.finishes += int(finish_win)
        if official_decision:
            self.result_history.append(1 if won else 0)
        self.strikes_att += stats.get("strikes_att", 0.0)
        self.strikes_succ += stats.get("strikes_succ", 0.0)
        self.strikes_att_against += opponent_stats.get("strikes_att", 0.0)
        self.strikes_succ_against += opponent_stats.get("strikes_succ", 0.0)
        self.total_strikes_att += stats.get("total_strikes_att", 0.0)
        self.total_strikes_succ += stats.get("total_strikes_succ", 0.0)
        self.total_strikes_att_against += opponent_stats.get("total_strikes_att", 0.0)
        self.total_strikes_succ_against += opponent_stats.get("total_strikes_succ", 0.0)
        self.knockdowns_for += stats.get("knockdowns", 0.0)
        self.knockdowns_against += opponent_stats.get("knockdowns", 0.0)
        self.takedown_att += stats.get("takedown_att", 0.0)
        self.takedown_succ += stats.get("takedown_succ", 0.0)
        self.takedown_att_against += opponent_stats.get("takedown_att", 0.0)
        self.takedown_succ_against += opponent_stats.get("takedown_succ", 0.0)
        self.submission_att += stats.get("submission_att", 0.0)
        self.reversals += stats.get("reversals", 0.0)
        self.ctrl_time_seconds += stats.get("ctrl_time_seconds", 0.0)
        self.ctrl_time_seconds_against += opponent_stats.get("ctrl_time_seconds", 0.0)
        self.fight_minutes += max(stats.get("fight_minutes", 15.0), 1.0)
        self.late_fight_minutes += stats.get("late_fight_minutes", 0.0)
        self.late_strikes_att += stats.get("late_strikes_att", 0.0)
        self.late_strikes_succ += stats.get("late_strikes_succ", 0.0)
        self.late_strikes_att_against += opponent_stats.get("late_strikes_att", 0.0)
        self.late_strikes_succ_against += opponent_stats.get("late_strikes_succ", 0.0)
        self.late_takedown_att += stats.get("late_takedown_att", 0.0)
        self.late_takedown_succ += stats.get("late_takedown_succ", 0.0)
        self.late_takedown_att_against += opponent_stats.get("late_takedown_att", 0.0)
        self.late_takedown_succ_against += opponent_stats.get("late_takedown_succ", 0.0)
        self.late_ctrl_time_seconds += stats.get("late_ctrl_time_seconds", 0.0)
        self.late_ctrl_time_seconds_against += opponent_stats.get("late_ctrl_time_seconds", 0.0)
        self.champ_fight_minutes += stats.get("champ_fight_minutes", 0.0)
        self.champ_strikes_succ += stats.get("champ_strikes_succ", 0.0)
        self.champ_strikes_succ_against += opponent_stats.get("champ_strikes_succ", 0.0)
        self.champ_ctrl_time_seconds += stats.get("champ_ctrl_time_seconds", 0.0)
        self.champ_ctrl_time_seconds_against += opponent_stats.get("champ_ctrl_time_seconds", 0.0)
        if opponent_elo is not None:
            self.opponent_elo_sum += opponent_elo
        if self.first_fight_date is None:
            self.first_fight_date = event_date
        self.last_fight_date = event_date


@dataclass
class FeatureBuilder:
    elo: EloTracker = field(default_factory=EloTracker)

    def build_training_frame(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        events, fighters, fights, fight_stats = _prepare_tables(tables)
        merged = fights.merge(events, on="event_id", how="left").sort_values(["event_date", "fight_id"])
        fighter_lookup = fighters.set_index("fighter_id").to_dict("index")
        stats_lookup = _stats_lookup(fight_stats)
        histories: dict[str, FighterHistory] = {}
        rows: list[dict[str, object]] = []

        for fight in merged.to_dict("records"):
            result = str(fight.get("result", "")).strip().lower()
            winner_id = fight.get("winner_id")
            if pd.isna(winner_id) or result in OUTCOME_EXCLUDE:
                _update_histories(histories, fight, stats_lookup, event_date=fight["event_date"], elo=self.elo)
                continue

            fighter_a_id = str(fight["fighter_a_id"])
            fighter_b_id = str(fight["fighter_b_id"])
            event_date = fight["event_date"]
            row = self._matchup_features(
                fight=fight,
                fighter_a=fighter_lookup[fighter_a_id],
                fighter_b=fighter_lookup[fighter_b_id],
                history_a=histories.get(fighter_a_id, FighterHistory()),
                history_b=histories.get(fighter_b_id, FighterHistory()),
                event_date=event_date,
            )
            row["target_fighter_a_win"] = int(str(winner_id) == fighter_a_id)
            row["data_quality_flags"] = _data_quality_flags(row)
            rows.append(row)
            _update_histories(histories, fight, stats_lookup, event_date=event_date, elo=self.elo)
            self.elo.update(fighter_a_id, fighter_b_id, bool(row["target_fighter_a_win"]))

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError("No trainable fights found after excluding draws/no contests.")
        return clean_feature_frame(frame)

    def build_prediction_frame(
        self,
        tables: dict[str, pd.DataFrame],
        upcoming_fights: pd.DataFrame,
    ) -> pd.DataFrame:
        events, fighters, fights, fight_stats = _prepare_tables(tables)
        historical = fights.merge(events, on="event_id", how="left").sort_values(["event_date", "fight_id"])
        histories: dict[str, FighterHistory] = {}
        stats_lookup = _stats_lookup(fight_stats)
        self.elo = EloTracker()

        for fight in historical.to_dict("records"):
            result = str(fight.get("result", "")).strip().lower()
            winner_id = fight.get("winner_id")
            _update_histories(histories, fight, stats_lookup, event_date=fight["event_date"], elo=self.elo)
            if pd.notna(winner_id) and result not in OUTCOME_EXCLUDE:
                self.elo.update(
                    str(fight["fighter_a_id"]),
                    str(fight["fighter_b_id"]),
                    str(winner_id) == str(fight["fighter_a_id"]),
                )

        fighters = fighters.copy()
        fighters["normalized_name"] = fighters["fighter_name"].map(normalize_name)
        fighter_by_name = fighters.set_index("normalized_name").to_dict("index")
        rows: list[dict[str, object]] = []
        for fight in _prepare_upcoming(upcoming_fights).to_dict("records"):
            name_a = normalize_name(fight["fighter_a"])
            name_b = normalize_name(fight["fighter_b"])
            if name_a not in fighter_by_name or name_b not in fighter_by_name:
                missing = [name for name in [fight["fighter_a"], fight["fighter_b"]] if normalize_name(name) not in fighter_by_name]
                raise KeyError(f"Upcoming fighter names not found in fighters.csv: {missing}")
            fighter_a = fighter_by_name[name_a]
            fighter_b = fighter_by_name[name_b]
            synthetic = {
                "fight_id": f"upcoming-{len(rows) + 1}",
                "event_id": "upcoming",
                "event_date": fight["event_date"],
                "fighter_a_id": fighter_a["fighter_id"],
                "fighter_b_id": fighter_b["fighter_id"],
                "weight_class": fight["weight_class"],
                "gender": fight["gender"],
                "scheduled_rounds": fight["scheduled_rounds"],
                "title_fight": fight["title_fight"],
            }
            row = self._matchup_features(
                fight=synthetic,
                fighter_a=fighter_a,
                fighter_b=fighter_b,
                history_a=histories.get(str(fighter_a["fighter_id"]), FighterHistory()),
                history_b=histories.get(str(fighter_b["fighter_id"]), FighterHistory()),
                event_date=fight["event_date"],
            )
            row["fighter_a"] = fight["fighter_a"]
            row["fighter_b"] = fight["fighter_b"]
            row["data_quality_flags"] = _data_quality_flags(row)
            rows.append(row)
        return clean_feature_frame(pd.DataFrame(rows), require_target=False)

    def _matchup_features(
        self,
        fight: dict[str, object],
        fighter_a: dict[str, object],
        fighter_b: dict[str, object],
        history_a: FighterHistory,
        history_b: FighterHistory,
        event_date: pd.Timestamp,
    ) -> dict[str, object]:
        snapshot_a = history_a.snapshot(event_date)
        snapshot_b = history_b.snapshot(event_date)
        fighter_a_id = str(fight["fighter_a_id"])
        fighter_b_id = str(fight["fighter_b_id"])
        row: dict[str, object] = {
            "fight_id": fight["fight_id"],
            "event_id": fight["event_id"],
            "event_date": event_date.date().isoformat(),
            "fighter_a_id": fighter_a_id,
            "fighter_b_id": fighter_b_id,
            "fighter_a_name": fighter_a["fighter_name"],
            "fighter_b_name": fighter_b["fighter_name"],
            "weight_class": fight["weight_class"],
            "gender": fight["gender"],
            "scheduled_rounds": float(fight["scheduled_rounds"]),
            "title_fight": float(bool(fight["title_fight"])),
            "fighter_a_stance": fighter_a.get("fighter_stance", "Unknown") or "Unknown",
            "fighter_b_stance": fighter_b.get("fighter_stance", "Unknown") or "Unknown",
        }
        row["same_stance"] = float(row["fighter_a_stance"] == row["fighter_b_stance"])
        row["age_diff"] = _age_years(fighter_a.get("fighter_dob"), event_date) - _age_years(fighter_b.get("fighter_dob"), event_date)
        row["height_diff_cm"] = _safe_float(fighter_a.get("fighter_height_cm")) - _safe_float(fighter_b.get("fighter_height_cm"))
        row["reach_diff_cm"] = _safe_float(fighter_a.get("fighter_reach_cm")) - _safe_float(fighter_b.get("fighter_reach_cm"))
        row["weight_diff_lbs"] = _safe_float(fighter_a.get("fighter_weight_lbs")) - _safe_float(fighter_b.get("fighter_weight_lbs"))
        row["elo_prob_fighter_a"] = self.elo.probability(fighter_a_id, fighter_b_id)
        row["elo_diff"] = self.elo.rating(fighter_a_id) - self.elo.rating(fighter_b_id)
        for key, value in snapshot_a.items():
            row[f"fighter_a_{key}"] = value
        for key, value in snapshot_b.items():
            row[f"fighter_b_{key}"] = value
        row["experience_diff"] = row["fighter_a_ufc_fights"] - row["fighter_b_ufc_fights"]
        row["loss_diff"] = row["fighter_a_ufc_losses"] - row["fighter_b_ufc_losses"]
        _add_history_diffs(row)
        return row


def build_features_from_raw(raw_dir: str | Path) -> pd.DataFrame:
    return FeatureBuilder().build_training_frame(load_raw_tables(raw_dir))


def clean_feature_frame(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in NUMERIC_MODEL_FEATURES:
        if column not in cleaned.columns:
            cleaned[column] = np.nan
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    median_fill = cleaned[NUMERIC_MODEL_FEATURES].median(numeric_only=True).fillna(0.0)
    cleaned[NUMERIC_MODEL_FEATURES] = cleaned[NUMERIC_MODEL_FEATURES].fillna(median_fill).fillna(0.0)
    for column in CATEGORICAL_MODEL_FEATURES:
        if column not in cleaned.columns:
            cleaned[column] = "Unknown"
        cleaned[column] = cleaned[column].fillna("Unknown").astype(str)
    if require_target and "target_fighter_a_win" not in cleaned.columns:
        raise ValueError("Feature frame is missing target_fighter_a_win.")
    return cleaned


def _prepare_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = tables["events"].copy()
    fighters = tables["fighters"].copy()
    fights = tables["fights"].copy()
    fight_stats = tables["fight_stats"].copy()

    events["event_id"] = events["event_id"].astype(str)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="raise")
    fighters["fighter_id"] = fighters["fighter_id"].astype(str)
    fighters["fighter_name"] = fighters["fighter_name"].astype(str)
    fights["fight_id"] = fights["fight_id"].astype(str)
    fights["event_id"] = fights["event_id"].astype(str)
    fights["fighter_a_id"] = fights["fighter_a_id"].astype(str)
    fights["fighter_b_id"] = fights["fighter_b_id"].astype(str)
    fights["winner_id"] = fights["winner_id"].astype("string")
    fights["scheduled_rounds"] = pd.to_numeric(fights["scheduled_rounds"], errors="coerce").fillna(3)
    fights["title_fight"] = fights["title_fight"].map(_parse_bool)
    fight_stats["fight_id"] = fight_stats["fight_id"].astype(str)
    fight_stats["fighter_id"] = fight_stats["fighter_id"].astype(str)
    if "fight_minutes" not in fight_stats.columns:
        fight_stats["fight_minutes"] = 15.0
    return events, fighters, fights, fight_stats


def _prepare_upcoming(upcoming: pd.DataFrame) -> pd.DataFrame:
    required = ["event_date", "fighter_a", "fighter_b", "weight_class", "gender", "scheduled_rounds", "title_fight"]
    missing = [column for column in required if column not in upcoming.columns]
    if missing:
        raise ValueError(f"Upcoming fight CSV missing columns: {missing}")
    prepared = upcoming.copy()
    prepared["event_date"] = pd.to_datetime(prepared["event_date"], errors="raise")
    prepared["scheduled_rounds"] = pd.to_numeric(prepared["scheduled_rounds"], errors="coerce").fillna(3)
    prepared["title_fight"] = prepared["title_fight"].map(_parse_bool)
    return prepared


def _stats_lookup(fight_stats: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    lookup: dict[tuple[str, str], dict[str, float]] = {}
    numeric_columns = [
        "knockdowns",
        "strikes_att",
        "strikes_succ",
        "total_strikes_att",
        "total_strikes_succ",
        "takedown_att",
        "takedown_succ",
        "submission_att",
        "reversals",
        "ctrl_time_seconds",
        "fight_minutes",
        "late_knockdowns",
        "late_strikes_att",
        "late_strikes_succ",
        "late_total_strikes_att",
        "late_total_strikes_succ",
        "late_takedown_att",
        "late_takedown_succ",
        "late_submission_att",
        "late_reversals",
        "late_ctrl_time_seconds",
        "late_fight_minutes",
        "champ_knockdowns",
        "champ_strikes_att",
        "champ_strikes_succ",
        "champ_total_strikes_att",
        "champ_total_strikes_succ",
        "champ_takedown_att",
        "champ_takedown_succ",
        "champ_submission_att",
        "champ_reversals",
        "champ_ctrl_time_seconds",
        "champ_fight_minutes",
    ]
    for row in fight_stats.to_dict("records"):
        lookup[(str(row["fight_id"]), str(row["fighter_id"]))] = {
            column: _safe_float(row.get(column, 0.0)) for column in numeric_columns
        }
    return lookup


def _update_histories(
    histories: dict[str, FighterHistory],
    fight: dict[str, object],
    stats_lookup: dict[tuple[str, str], dict[str, float]],
    event_date: pd.Timestamp,
    elo: EloTracker | None = None,
) -> None:
    result = str(fight.get("result", "")).strip().lower()
    winner_id = str(fight.get("winner_id")) if pd.notna(fight.get("winner_id")) else None
    is_finish = result in {"ko/tko", "submission", "dq"}
    official_decision = winner_id is not None and result not in OUTCOME_EXCLUDE
    fighter_a_id = str(fight["fighter_a_id"])
    fighter_b_id = str(fight["fighter_b_id"])
    stats_a = stats_lookup.get((str(fight["fight_id"]), fighter_a_id), {})
    stats_b = stats_lookup.get((str(fight["fight_id"]), fighter_b_id), {})
    for fighter_id, opponent_id, stats, opponent_stats in [
        (fighter_a_id, fighter_b_id, stats_a, stats_b),
        (fighter_b_id, fighter_a_id, stats_b, stats_a),
    ]:
        histories.setdefault(fighter_id, FighterHistory())
        won = winner_id == fighter_id
        histories[fighter_id].update(
            won=won,
            finish_win=won and is_finish,
            stats=stats,
            opponent_stats=opponent_stats,
            event_date=event_date,
            opponent_elo=elo.rating(opponent_id) if elo is not None else None,
            official_decision=official_decision,
        )


def _age_years(dob: object, event_date: pd.Timestamp) -> float:
    parsed = pd.to_datetime(dob, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    return float((event_date - parsed).days / 365.25)


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _recent_win_rate(results: list[int], window: int) -> float:
    if not results:
        return 0.5
    recent = results[-window:]
    return float(sum(recent) / len(recent))


def _current_streak(results: list[int], outcome: int) -> int:
    streak = 0
    for result in reversed(results):
        if result != outcome:
            break
        streak += 1
    return streak


def _rate_per(value: float, minutes: float, per_minutes: float = 5.0) -> float:
    if minutes <= 0:
        return 0.0
    return value / minutes * per_minutes


def _add_history_diffs(row: dict[str, object]) -> None:
    diff_specs = {
        "win_rate": "win_rate_diff",
        "recent_win_rate_3": "recent_win_rate_3_diff",
        "recent_win_rate_5": "recent_win_rate_5_diff",
        "current_win_streak": "current_win_streak_diff",
        "current_loss_streak": "current_loss_streak_diff",
        "finish_rate": "finish_rate_diff",
        "layoff_days": "layoff_diff_days",
        "days_since_ufc_debut": "days_since_ufc_debut_diff",
        "avg_fight_minutes": "avg_fight_minutes_diff",
        "late_round_minutes": "late_round_minutes_diff",
        "late_round_share": "late_round_share_diff",
        "champ_round_minutes": "champ_round_minutes_diff",
        "avg_opponent_elo": "avg_opponent_elo_diff",
        "strikes_landed_per_15": "strikes_landed_per_15_diff",
        "strikes_absorbed_per_15": "strikes_absorbed_per_15_diff",
        "strike_differential_per_15": "strike_differential_per_15_diff",
        "strike_accuracy": "strike_accuracy_diff",
        "strike_defense": "strike_defense_diff",
        "total_strikes_landed_per_15": "total_strikes_landed_per_15_diff",
        "total_strikes_absorbed_per_15": "total_strikes_absorbed_per_15_diff",
        "total_strike_differential_per_15": "total_strike_differential_per_15_diff",
        "knockdowns_for_per_15": "knockdowns_for_per_15_diff",
        "knockdowns_against_per_15": "knockdowns_against_per_15_diff",
        "knockdown_differential_per_15": "knockdown_differential_per_15_diff",
        "takedowns_per_15": "takedowns_per_15_diff",
        "takedown_accuracy": "takedown_accuracy_diff",
        "takedowns_absorbed_per_15": "takedowns_absorbed_per_15_diff",
        "takedown_defense": "takedown_defense_diff",
        "takedown_differential_per_15": "takedown_differential_per_15_diff",
        "submission_attempts_per_15": "submission_attempts_per_15_diff",
        "control_seconds_per_15": "control_seconds_per_15_diff",
        "control_absorbed_seconds_per_15": "control_absorbed_seconds_per_15_diff",
        "control_differential_seconds_per_15": "control_differential_seconds_per_15_diff",
        "late_strikes_landed_per_5": "late_strikes_landed_per_5_diff",
        "late_strikes_absorbed_per_5": "late_strikes_absorbed_per_5_diff",
        "late_strike_differential_per_5": "late_strike_differential_per_5_diff",
        "late_strike_accuracy": "late_strike_accuracy_diff",
        "late_strike_defense": "late_strike_defense_diff",
        "late_takedowns_per_5": "late_takedowns_per_5_diff",
        "late_takedown_defense": "late_takedown_defense_diff",
        "late_control_differential_seconds_per_5": "late_control_differential_seconds_per_5_diff",
        "champ_strike_differential_per_5": "champ_strike_differential_per_5_diff",
        "champ_control_differential_seconds_per_5": "champ_control_differential_seconds_per_5_diff",
    }
    for base, diff in diff_specs.items():
        row[diff] = row[f"fighter_a_{base}"] - row[f"fighter_b_{base}"]


def _data_quality_flags(row: dict[str, object]) -> str:
    flags: list[str] = []
    if row.get("fighter_a_ufc_fights", 0) == 0:
        flags.append("fighter_a_debut_or_missing_history")
    if row.get("fighter_b_ufc_fights", 0) == 0:
        flags.append("fighter_b_debut_or_missing_history")
    layoff_a = row.get("fighter_a_layoff_days")
    layoff_b = row.get("fighter_b_layoff_days")
    if pd.isna(layoff_a) or pd.isna(layoff_b):
        flags.append("layoff_imputed")
    return ";".join(flags)
