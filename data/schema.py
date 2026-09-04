"""Column definitions and label semantics of the KuaiRand interaction logs.

Field semantics follow the official documentation at https://kuairand.com/.
The thresholds below are part of the dataset definition, not tunable knobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The logs were collected in China; only this timezone reproduces the
# date / hourmin columns from time_ms.
DATASET_TZ = "Asia/Shanghai"

# is_click equals valid-play in the single-column UI: videos up to this length
# must be watched to the end, longer ones must be watched past it.
CLICK_DURATION_THRESHOLD_MS = 7_000

# Same idea for long_view, with an 18s threshold.
LONG_VIEW_DURATION_THRESHOLD_MS = 18_000

# Explicit dtypes keep a 1.4M-row log at ~80MB instead of several hundred.
LOG_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "date": "int32",
    "hourmin": "int32",
    "time_ms": "int64",
    "is_click": "int8",
    "is_like": "int8",
    "is_follow": "int8",
    "is_comment": "int8",
    "is_forward": "int8",
    "is_hate": "int8",
    "long_view": "int8",
    "play_time_ms": "int32",
    "duration_ms": "int32",
    "profile_stay_time": "int32",
    "comment_stay_time": "int32",
    "is_profile_enter": "int8",
    "is_rand": "int8",
    "tab": "int8",
}

# Fields Stage 0 must confirm are present.
REQUIRED_COLUMNS = (
    "user_id",
    "video_id",
    "time_ms",
    "is_click",
    "is_like",
    "is_follow",
    "is_forward",
    "is_hate",
    "long_view",
    "play_time_ms",
    "duration_ms",
    "is_rand",
)

# Columns that must only ever contain 0/1.
BINARY_COLUMNS = (
    "is_click",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "long_view",
    "is_profile_enter",
    "is_rand",
)

# Feedback signals, ordered by density.
FEEDBACK_COLUMNS = (
    "is_click",
    "long_view",
    "is_like",
    "is_profile_enter",
    "is_comment",
    "is_follow",
    "is_forward",
    "is_hate",
)


class SchemaError(ValueError):
    """Raised when a log does not match the expected KuaiRand schema."""


def valid_play(play_time_ms: pd.Series, duration_ms: pd.Series) -> np.ndarray:
    """Recompute the official valid-play rule behind is_click."""
    play = play_time_ms.to_numpy()
    duration = duration_ms.to_numpy()
    short = duration <= CLICK_DURATION_THRESHOLD_MS
    return np.where(short, play >= duration, play > CLICK_DURATION_THRESHOLD_MS)


def long_view(play_time_ms: pd.Series, duration_ms: pd.Series) -> np.ndarray:
    """Recompute the official rule behind long_view."""
    play = play_time_ms.to_numpy()
    duration = duration_ms.to_numpy()
    short = duration <= LONG_VIEW_DURATION_THRESHOLD_MS
    return np.where(short, play >= duration, play >= LONG_VIEW_DURATION_THRESHOLD_MS)


def validate_log(df: pd.DataFrame) -> None:
    """Fail loudly on missing required fields, nulls, or non-binary labels."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        raise SchemaError(f"nulls in required columns: {nulls.to_dict()}")

    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue
        values = set(pd.unique(df[column]).tolist())
        if not values <= {0, 1}:
            raise SchemaError(f"column {column} is not binary: {sorted(values)}")


def add_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Attach a timezone-aware timestamp derived from time_ms (in place)."""
    df[column] = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert(DATASET_TZ)
    return df
