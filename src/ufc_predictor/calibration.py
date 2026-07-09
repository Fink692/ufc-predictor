from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(y_true, probabilities, bins: int = 10) -> pd.DataFrame:
    if bins <= 0:
        raise ValueError("bins must be positive.")
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have the same length.")
    if len(y) == 0:
        return pd.DataFrame(
            columns=[
                "bucket",
                "low",
                "high",
                "row_count",
                "average_predicted_probability",
                "observed_win_rate",
                "calibration_gap",
                "absolute_calibration_gap",
            ]
        )

    clipped = np.clip(probs, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket_ids = np.clip(np.digitize(clipped, edges, right=False) - 1, 0, bins - 1)
    rows: list[dict[str, object]] = []
    for bucket_id in range(bins):
        mask = bucket_ids == bucket_id
        low = float(edges[bucket_id])
        high = float(edges[bucket_id + 1])
        if not mask.any():
            rows.append(
                {
                    "bucket": f"{low:.0%}-{high:.0%}",
                    "low": low,
                    "high": high,
                    "row_count": 0,
                    "average_predicted_probability": None,
                    "observed_win_rate": None,
                    "calibration_gap": None,
                    "absolute_calibration_gap": None,
                }
            )
            continue
        avg_predicted = float(clipped[mask].mean())
        observed = float(y[mask].mean())
        gap = observed - avg_predicted
        rows.append(
            {
                "bucket": f"{low:.0%}-{high:.0%}",
                "low": low,
                "high": high,
                "row_count": int(mask.sum()),
                "average_predicted_probability": avg_predicted,
                "observed_win_rate": observed,
                "calibration_gap": gap,
                "absolute_calibration_gap": abs(gap),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(y_true, probabilities, bins: int = 10) -> dict[str, float | int | None]:
    table = calibration_table(y_true, probabilities, bins=bins)
    total = int(table["row_count"].sum()) if not table.empty else 0
    if total == 0:
        return {
            "row_count": 0,
            "bins": bins,
            "expected_calibration_error": None,
            "maximum_calibration_error": None,
        }
    populated = table[table["row_count"] > 0].copy()
    weights = populated["row_count"] / total
    ece = float((weights * populated["absolute_calibration_gap"]).sum())
    mce = float(populated["absolute_calibration_gap"].max())
    return {
        "row_count": total,
        "bins": bins,
        "expected_calibration_error": ece,
        "maximum_calibration_error": mce,
    }
