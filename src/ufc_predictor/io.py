from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .config import REQUIRED_RAW_FILES


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    ensure_parent(output)
    df.to_csv(output, index=False)


def write_json(data: Mapping, path: str | Path) -> None:
    import json

    output = Path(path)
    ensure_parent(output)
    output.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_raw_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for table_name, file_name in REQUIRED_RAW_FILES.items():
        csv_path = raw_path / file_name
        if not csv_path.exists():
            missing.append(str(csv_path))
            continue
        tables[table_name] = pd.read_csv(csv_path)
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing required raw CSV files:\n  - {joined}")
    return tables


def normalize_name(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())
