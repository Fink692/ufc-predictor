from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(".")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_path: Path = Path("models/ufc_model.joblib")
    reports_dir: Path = Path("reports")


REQUIRED_RAW_FILES = {
    "events": "events.csv",
    "fighters": "fighters.csv",
    "fights": "fights.csv",
    "fight_stats": "fight_stats.csv",
}
