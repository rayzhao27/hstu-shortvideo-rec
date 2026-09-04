"""Train / val / test assignment, and the leakage check that justifies the choice.

Two strategies:

``temporal`` (default)
    Global time cut: the last ``test_days`` days of the dataset are test, the
    ``val_days`` before them are val, everything earlier is train. This mirrors
    production - the model only ever sees the past - and makes future leakage
    impossible by construction, since every train interaction precedes every val
    interaction, which precedes every test interaction.

``loo``
    Leave-last-one-out per user: a user's last interaction is test, the previous one
    is val. This is what SASRec / BERT4Rec / most sequential-rec papers report, so it
    is here for comparability, but it *does* leak time across users: an active user's
    train interactions can happen after a sparse user's test interaction, so the model
    trains on a future the evaluation pretends it cannot see.

:func:`leakage_report` quantifies exactly that, so the cost of picking ``loo`` is a
number in ``preprocess_stats.json`` rather than a footnote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.schema import DATASET_TZ

logger = logging.getLogger(__name__)

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
SPLIT_CODES: dict[str, int] = {"train": TRAIN, "val": VAL, "test": TEST}

_MS_PER_DAY = 86_400_000


@dataclass
class SplitResult:
    """Per-row split codes plus the metadata needed to explain them."""

    codes: np.ndarray
    strategy: str
    boundaries: dict = field(default_factory=dict)

    def counts(self) -> dict:
        bins = np.bincount(self.codes, minlength=3)
        total = int(bins.sum())
        return {
            name: {"n_interactions": int(bins[code]), "share": float(bins[code] / total)}
            for name, code in SPLIT_CODES.items()
        }


def _as_local(time_ms: int) -> str:
    return pd.Timestamp(time_ms, unit="ms", tz="UTC").tz_convert(DATASET_TZ).isoformat()


def temporal_split(df: pd.DataFrame, val_days: float = 3.0, test_days: float = 4.0) -> SplitResult:
    """Cut the global timeline into train / val / test by wall-clock time."""
    time_ms = df["time_ms"].to_numpy()
    t_max = int(time_ms.max())
    test_start = t_max - int(test_days * _MS_PER_DAY)
    val_start = test_start - int(val_days * _MS_PER_DAY)

    if val_start <= int(time_ms.min()):
        raise ValueError(
            f"val_days={val_days} + test_days={test_days} covers the whole "
            f"{(t_max - int(time_ms.min())) / _MS_PER_DAY:.1f}-day dataset, leaving no train data"
        )

    codes = np.full(len(df), TRAIN, dtype=np.int8)
    codes[time_ms >= val_start] = VAL
    codes[time_ms >= test_start] = TEST

    logger.info(
        "temporal split: train < %s <= val < %s <= test",
        _as_local(val_start)[:16],
        _as_local(test_start)[:16],
    )
    return SplitResult(
        codes=codes,
        strategy="temporal",
        boundaries={
            "val_start": _as_local(val_start),
            "test_start": _as_local(test_start),
            "val_days": val_days,
            "test_days": test_days,
        },
    )


def loo_split(df: pd.DataFrame, n_val: int = 1, n_test: int = 1) -> SplitResult:
    """Leave the last ``n_test`` interactions of each user for test, previous for val.

    Expects ``df`` sorted by (user_id, time_ms).
    """
    rank_from_end = df.groupby("user_id", sort=False).cumcount(ascending=False).to_numpy()

    codes = np.full(len(df), TRAIN, dtype=np.int8)
    codes[rank_from_end < n_test + n_val] = VAL
    codes[rank_from_end < n_test] = TEST

    logger.info("leave-last-one-out split: %d test, %d val per user", n_test, n_val)
    return SplitResult(
        codes=codes,
        strategy="loo",
        boundaries={"n_val_per_user": n_val, "n_test_per_user": n_test},
    )


def make_split(df: pd.DataFrame, strategy: str = "temporal", **kwargs) -> SplitResult:
    if strategy == "temporal":
        return temporal_split(df, **kwargs)
    if strategy == "loo":
        return loo_split(df, **kwargs)
    raise ValueError(f"unknown split strategy {strategy!r}, expected 'temporal' or 'loo'")


def leakage_report(df: pd.DataFrame, split: SplitResult) -> dict:
    """Measure how much of train/val happens after the evaluation windows start.

    Zero for a temporal split. For leave-last-one-out it is large, which is the
    honest cost of that strategy.
    """
    time_ms = df["time_ms"].to_numpy()
    report: dict = {"strategy": split.strategy, "time_ranges": {}}

    for name, code in SPLIT_CODES.items():
        mask = split.codes == code
        if not mask.any():
            report["time_ranges"][name] = None
            continue
        report["time_ranges"][name] = {
            "start": _as_local(int(time_ms[mask].min())),
            "end": _as_local(int(time_ms[mask].max())),
        }

    train_mask, val_mask, test_mask = (split.codes == c for c in (TRAIN, VAL, TEST))
    if test_mask.any():
        test_start = int(time_ms[test_mask].min())
        report["train_after_test_start"] = {
            "n": int((time_ms[train_mask] >= test_start).sum()),
            "share_of_train": float((time_ms[train_mask] >= test_start).mean()),
        }
    if val_mask.any():
        val_start = int(time_ms[val_mask].min())
        report["train_after_val_start"] = {
            "n": int((time_ms[train_mask] >= val_start).sum()),
            "share_of_train": float((time_ms[train_mask] >= val_start).mean()),
        }

    report["leak_free"] = (
        report.get("train_after_test_start", {}).get("n", 0) == 0
        and report.get("train_after_val_start", {}).get("n", 0) == 0
    )
    return report
