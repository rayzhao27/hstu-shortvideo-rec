"""Read KuaiRand CSV logs and feature tables."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.schema import LOG_DTYPES, add_timestamp

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("datasets/processed")

# Globs rather than hardcoded names, so the same code works for the Pure / 1K /
# 27K releases (they differ by filename suffix, and 27K splits logs into parts).
LOG_GLOBS = {
    "standard": "log_standard_*.csv",
    "random": "log_random_*.csv",
}

FEATURE_GLOBS = {
    "user_features": "user_features_*.csv",
    "video_features_basic": "video_features_basic_*.csv",
    "video_features_statistic": "video_features_statistic_*.csv",
}

SPLITS = tuple(LOG_GLOBS)


def read_log_csv(path: Path) -> pd.DataFrame:
    """Read one log CSV, applying the known dtypes to the columns it has."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtypes = {c: LOG_DTYPES[c] for c in header if c in LOG_DTYPES}
    unknown = [c for c in header if c not in LOG_DTYPES]
    if unknown:
        logger.warning("%s has columns absent from LOG_DTYPES: %s", path.name, unknown)
    return pd.read_csv(path, dtype=dtypes)


def load_log(
    data_dir: Path,
    split: str = "standard",
    with_timestamp: bool = True,
    sort: bool = True,
) -> pd.DataFrame:
    """Load and concatenate every CSV of one split.

    The standard split spans two files covering four consecutive weeks of the
    same users; concatenating and sorting by (user_id, time_ms) yields the user
    behaviour sequences the model consumes.
    """
    if split not in LOG_GLOBS:
        raise KeyError(f"unknown split {split!r}, expected one of {SPLITS}")

    paths = sorted(data_dir.glob(LOG_GLOBS[split]))
    if not paths:
        raise FileNotFoundError(f"no {split} log matching {LOG_GLOBS[split]!r} in {data_dir}")

    frames = []
    for path in paths:
        frame = read_log_csv(path)
        logger.info("  %-42s rows=%9d users=%6d", path.name, len(frame), frame.user_id.nunique())
        frames.append(frame)

    log = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if sort:
        log = log.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)
    if with_timestamp:
        add_timestamp(log)

    logger.info(
        "%s log: %d rows x %d cols, %.1f MB in memory",
        split,
        len(log),
        log.shape[1],
        log.memory_usage(deep=True).sum() / 1024**2,
    )
    return log


def load_features(data_dir: Path, name: str) -> pd.DataFrame:
    """Load one of the feature tables listed in FEATURE_GLOBS."""
    if name not in FEATURE_GLOBS:
        raise KeyError(f"unknown feature table {name!r}, expected {list(FEATURE_GLOBS)}")

    matches = sorted(data_dir.glob(FEATURE_GLOBS[name]))
    if not matches:
        raise FileNotFoundError(f"no file matching {FEATURE_GLOBS[name]!r} in {data_dir}")
    if len(matches) > 1:
        raise FileNotFoundError(f"{FEATURE_GLOBS[name]!r} is ambiguous in {data_dir}: {matches}")

    frame = pd.read_csv(matches[0])
    logger.info("%-24s %6d rows x %3d cols", name, len(frame), frame.shape[1])
    return frame


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Cache a table as parquet, dropping the derived timestamp column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = df.drop(columns=["timestamp"]) if "timestamp" in df.columns else df
    frame.to_parquet(path, index=False)
    logger.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path


def read_parquet(path: Path, with_timestamp: bool = True) -> pd.DataFrame:
    """Read a cached table back, re-deriving the timestamp column."""
    frame = pd.read_parquet(path)
    if with_timestamp and "time_ms" in frame.columns:
        add_timestamp(frame)
    return frame
