from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .io import write_csv, write_json


DEFAULT_MIRROR_BASE_URL = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main"

SOURCE_FILES = {
    "events": "ufc_event_details.csv",
    "fight_results": "ufc_fight_results.csv",
    "fight_stats": "ufc_fight_stats.csv",
    "fighters": "ufc_fighter_tott.csv",
}

KNOWN_WEIGHT_CLASSES = [
    "Light Heavyweight",
    "Open Weight",
    "Catch Weight",
    "Strawweight",
    "Flyweight",
    "Bantamweight",
    "Featherweight",
    "Lightweight",
    "Welterweight",
    "Middleweight",
    "Heavyweight",
    "Superfight",
]


def convert_ufcstats_mirror(output_raw_dir: str | Path, base_url: str = DEFAULT_MIRROR_BASE_URL) -> dict[str, object]:
    source = _load_source_tables(base_url)
    events = _convert_events(source["events"])
    fighters = _convert_fighters(source["fighters"])
    fighter_lookup = _fighter_lookup(fighters)
    fights = _convert_fights(source["fight_results"], events, fighter_lookup)
    fighters = _append_missing_fighters(fighters, fights)
    fight_stats = _convert_fight_stats(source["fight_stats"], fights, fighter_lookup)

    output = Path(output_raw_dir)
    write_csv(events, output / "events.csv")
    write_csv(fighters, output / "fighters.csv")
    write_csv(fights, output / "fights.csv")
    write_csv(fight_stats, output / "fight_stats.csv")

    metadata = {
        "source": base_url,
        "source_files": SOURCE_FILES,
        "events": int(len(events)),
        "fighters": int(len(fighters)),
        "fights": int(len(fights)),
        "fight_stats_rows": int(len(fight_stats)),
        "first_event_date": str(events["event_date"].min()),
        "latest_event_date": str(events["event_date"].max()),
        "trainable_fights_before_feature_exclusions": int(fights["winner_id"].notna().sum()),
    }
    write_json(metadata, output / "source_metadata.json")
    return metadata


def _load_source_tables(base_url: str) -> dict[str, pd.DataFrame]:
    base = base_url.rstrip("/")
    return {
        table_name: pd.read_csv(f"{base}/{file_name}")
        for table_name, file_name in SOURCE_FILES.items()
    }


def _convert_events(events: pd.DataFrame) -> pd.DataFrame:
    converted = events.copy()
    converted["event_id"] = converted["URL"].map(_id_from_url)
    converted["event_name"] = converted["EVENT"].astype(str).str.strip()
    converted["event_date"] = pd.to_datetime(converted["DATE"], errors="coerce").dt.date.astype(str)
    location_parts = converted["LOCATION"].fillna("").astype(str).str.split(",", expand=True)
    converted["event_city"] = location_parts.get(0, "").fillna("").str.strip()
    converted["event_state"] = location_parts.get(1, "").fillna("").str.strip()
    converted["event_country"] = location_parts.get(2, "").fillna("").str.strip()
    return converted[
        ["event_id", "event_date", "event_name", "event_city", "event_state", "event_country"]
    ].sort_values("event_date")


def _convert_fighters(fighters: pd.DataFrame) -> pd.DataFrame:
    converted = fighters.copy()
    converted["fighter_id"] = converted["URL"].map(_id_from_url)
    converted["fighter_name"] = converted["FIGHTER"].astype(str).str.strip()
    converted["fighter_dob"] = pd.to_datetime(converted["DOB"].replace("--", np.nan), errors="coerce").dt.date.astype(str)
    converted["fighter_height_cm"] = converted["HEIGHT"].map(_height_to_cm)
    converted["fighter_weight_lbs"] = converted["WEIGHT"].map(_pounds)
    converted["fighter_reach_cm"] = converted["REACH"].map(_inches_to_cm)
    converted["fighter_stance"] = converted["STANCE"].fillna("Unknown").replace("--", "Unknown")
    return converted[
        [
            "fighter_id",
            "fighter_name",
            "fighter_dob",
            "fighter_height_cm",
            "fighter_weight_lbs",
            "fighter_reach_cm",
            "fighter_stance",
        ]
    ].drop_duplicates("fighter_id")


def _append_missing_fighters(fighters: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    known_ids = set(fighters["fighter_id"].astype(str))
    rows: list[dict[str, object]] = []
    for fight in fights.to_dict("records"):
        fighter_a, fighter_b = _split_bout(fight["source_bout"])
        for fighter_id, fighter_name in [
            (str(fight["fighter_a_id"]), fighter_a),
            (str(fight["fighter_b_id"]), fighter_b),
        ]:
            if fighter_id not in known_ids:
                known_ids.add(fighter_id)
                rows.append(
                    {
                        "fighter_id": fighter_id,
                        "fighter_name": fighter_name,
                        "fighter_dob": "",
                        "fighter_height_cm": np.nan,
                        "fighter_weight_lbs": np.nan,
                        "fighter_reach_cm": np.nan,
                        "fighter_stance": "Unknown",
                    }
                )
    if not rows:
        return fighters
    return pd.concat([fighters, pd.DataFrame(rows)], ignore_index=True)


def _convert_fights(
    results: pd.DataFrame,
    events: pd.DataFrame,
    fighter_lookup: dict[str, str],
) -> pd.DataFrame:
    event_lookup = events.set_index(events["event_name"].map(_norm))["event_id"].to_dict()
    rows: list[dict[str, object]] = []
    for raw in results.to_dict("records"):
        fighter_a, fighter_b = _split_bout(raw["BOUT"])
        outcome = str(raw.get("OUTCOME", "")).strip().upper()
        outcome_a, outcome_b = _split_outcome(outcome)
        fighter_a_id = _lookup_fighter_id(fighter_a, fighter_lookup)
        fighter_b_id = _lookup_fighter_id(fighter_b, fighter_lookup)
        winner_id = np.nan
        result = _result_label(raw.get("METHOD"), outcome)
        if outcome_a == "W" and outcome_b == "L":
            winner_id = fighter_a_id
        elif outcome_a == "L" and outcome_b == "W":
            winner_id = fighter_b_id
        rows.append(
            {
                "fight_id": _id_from_url(raw["URL"]),
                "event_id": event_lookup[_norm(raw["EVENT"])],
                "source_event": str(raw["EVENT"]).strip(),
                "source_bout": str(raw["BOUT"]).strip(),
                "fighter_a_id": fighter_a_id,
                "fighter_b_id": fighter_b_id,
                "winner_id": winner_id,
                "weight_class": _weight_class(raw.get("WEIGHTCLASS")),
                "gender": "F" if "women" in str(raw.get("WEIGHTCLASS", "")).lower() else "M",
                "scheduled_rounds": _scheduled_rounds(raw.get("TIME FORMAT")),
                "title_fight": _is_title_fight(raw.get("WEIGHTCLASS")),
                "result": result,
                "finish_round": raw.get("ROUND"),
                "finish_time": raw.get("TIME"),
                "time_format": raw.get("TIME FORMAT"),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("fight_id")


def _convert_fight_stats(
    stats: pd.DataFrame,
    fights: pd.DataFrame,
    fighter_lookup: dict[str, str],
) -> pd.DataFrame:
    fight_id_by_event_bout = _fight_id_by_event_bout(fights)
    stats = stats.copy()
    stats["fighter_id"] = stats["FIGHTER"].map(lambda value: _lookup_fighter_id(value, fighter_lookup))
    stats["round_number"] = stats["ROUND"].map(_round_number)
    stats[["strikes_succ", "strikes_att"]] = stats["SIG.STR."].apply(_of_pair).apply(pd.Series)
    stats[["total_strikes_succ", "total_strikes_att"]] = stats["TOTAL STR."].apply(_of_pair).apply(pd.Series)
    stats[["takedown_succ", "takedown_att"]] = stats["TD"].apply(_of_pair).apply(pd.Series)
    stats["knockdowns"] = pd.to_numeric(stats["KD"], errors="coerce").fillna(0)
    stats["submission_att"] = pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0)
    stats["reversals"] = pd.to_numeric(stats["REV."], errors="coerce").fillna(0)
    stats["ctrl_time_seconds"] = stats["CTRL"].map(_clock_to_seconds)

    stats["fight_id"] = stats.apply(
        lambda row: fight_id_by_event_bout.get((_norm(row["EVENT"]), _norm(row["BOUT"]))),
        axis=1,
    )
    valid_stats = stats.dropna(subset=["fight_id", "fighter_id"])
    grouped = _group_stats(valid_stats)
    late_grouped = _group_stats(valid_stats[valid_stats["round_number"] >= 3], prefix="late_")
    champ_grouped = _group_stats(valid_stats[valid_stats["round_number"] >= 4], prefix="champ_")
    grouped = grouped.merge(late_grouped, on=["fight_id", "fighter_id"], how="left")
    grouped = grouped.merge(champ_grouped, on=["fight_id", "fighter_id"], how="left")
    grouped = grouped.fillna(0)
    duration_by_fight = fights.set_index("fight_id").apply(_fight_minutes_from_result, axis=1).to_dict()
    grouped["fight_minutes"] = grouped["fight_id"].map(duration_by_fight).fillna(15.0)
    grouped["late_fight_minutes"] = grouped["fight_minutes"].map(lambda minutes: max(float(minutes) - 10.0, 0.0))
    grouped["champ_fight_minutes"] = grouped["fight_minutes"].map(lambda minutes: max(float(minutes) - 15.0, 0.0))
    return grouped


def _group_stats(stats: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    numeric = [
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
    ]
    grouped = stats.groupby(["fight_id", "fighter_id"], as_index=False).agg({column: "sum" for column in numeric})
    if prefix:
        grouped = grouped.rename(columns={column: f"{prefix}{column}" for column in numeric})
    return grouped


def _fight_id_by_event_bout(fights: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (_norm(row["source_event"]), _norm(row["source_bout"])): row["fight_id"]
        for row in fights.to_dict("records")
    }


def _id_from_url(url: object) -> str:
    path = urlparse(str(url)).path.rstrip("/")
    return path.split("/")[-1]


def _split_bout(value: object) -> tuple[str, str]:
    text = str(value).strip()
    parts = re.split(r"\s+vs\.?\s+", text)
    if len(parts) != 2:
        raise ValueError(f"Could not split bout: {text}")
    return parts[0].strip(), parts[1].strip()


def _split_outcome(value: str) -> tuple[str, str]:
    parts = str(value).strip().upper().split("/")
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _fighter_lookup(fighters: pd.DataFrame) -> dict[str, str]:
    lookup = fighters.set_index(fighters["fighter_name"].map(_norm))["fighter_id"].to_dict()
    return lookup


def _lookup_fighter_id(name: object, fighter_lookup: dict[str, str]) -> str:
    key = _norm(name)
    if key in fighter_lookup:
        return fighter_lookup[key]
    fallback = re.sub(r"[^a-z0-9]+", "-", key).strip("-")
    return f"name-{fallback}"


def _norm(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _height_to_cm(value: object) -> float:
    text = str(value)
    match = re.search(r"(\d+)'\s*(\d+)", text)
    if not match:
        return np.nan
    inches = int(match.group(1)) * 12 + int(match.group(2))
    return round(inches * 2.54, 2)


def _inches_to_cm(value: object) -> float:
    text = str(value)
    match = re.search(r"(\d+)", text)
    if not match:
        return np.nan
    return round(int(match.group(1)) * 2.54, 2)


def _pounds(value: object) -> float:
    text = str(value)
    match = re.search(r"(\d+)", text)
    if not match:
        return np.nan
    return float(match.group(1))


def _result_label(method: object, outcome: str) -> str:
    if outcome == "D/D":
        return "Draw"
    if outcome == "NC/NC":
        return "No Contest"
    text = str(method).strip()
    lower = text.lower()
    if lower.startswith("decision"):
        return "Decision"
    if "submission" in lower:
        return "Submission"
    if "ko" in lower or "tko" in lower:
        return "KO/TKO"
    if lower == "dq":
        return "DQ"
    if "overturned" in lower:
        return "Overturned"
    return text or "Unknown"


def _weight_class(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value).replace("TitleBout", "Title Bout")).strip()
    for weight_class in KNOWN_WEIGHT_CLASSES:
        if weight_class.lower() in text.lower():
            return weight_class
    return text.replace(" Bout", "").strip() or "Unknown"


def _is_title_fight(value: object) -> bool:
    text = str(value).replace("TitleBout", "Title Bout").lower()
    return "title bout" in text or "championship bout" in text


def _scheduled_rounds(value: object) -> int:
    match = re.search(r"(\d+)\s+Rnd", str(value))
    if match:
        return int(match.group(1))
    return 3


def _of_pair(value: object) -> tuple[int, int]:
    match = re.search(r"(\d+)\s+of\s+(\d+)", str(value))
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _clock_to_seconds(value: object) -> int:
    text = str(value).strip()
    if text in {"", "--", "---", "nan"}:
        return 0
    match = re.search(r"(\d+):(\d+)", text)
    if not match:
        return 0
    return int(match.group(1)) * 60 + int(match.group(2))


def _round_number(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        return 0
    return int(match.group(1))


def _fight_minutes_from_result(row: pd.Series) -> float:
    finish_round = pd.to_numeric(row.get("finish_round"), errors="coerce")
    if pd.isna(finish_round) or finish_round < 1:
        return 15.0
    seconds = _clock_to_seconds(row.get("finish_time"))
    total_seconds = max((int(finish_round) - 1) * 300 + seconds, 60)
    return round(total_seconds / 60.0, 3)
