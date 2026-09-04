"""Exposure-bias bookkeeping around the ``is_rand`` flag.

KuaiRand ships two disjoint logs: ``log_standard``, everything the production
recommender chose to show (``is_rand=0``), and ``log_random``, videos injected
uniformly at random into the same users' feeds (``is_rand=1``). The random slice is
the reason to use this dataset at all - it is an unbiased sample of user response,
so it can measure what a model trained on logged feedback actually learned versus
what it inherited from the old policy.

Two facts checked on the real data justify merging them into one timeline:

* they never share a ``(user_id, video_id, time_ms)`` key, so merging cannot
  duplicate an impression;
* random exposures only exist for 2022-04-22..05-08, while standard exposures start
  2022-04-09, so the random share is not uniform over time and has to be reported
  per split rather than globally.

Keeping both in one chronological sequence reflects what the user actually saw. The
``is_rand`` flag rides along per interaction so Stage 3 can restrict evaluation to
the unbiased subset.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data.actions import action_distribution
from data.splitting import SPLIT_CODES

logger = logging.getLogger(__name__)

KEY_COLUMNS = ("user_id", "video_id", "time_ms")


def assert_logs_disjoint(standard: pd.DataFrame, random_log: pd.DataFrame) -> dict:
    """Confirm the two logs describe different impressions before merging."""
    key = list(KEY_COLUMNS)
    shared = standard[key].merge(random_log[key], on=key, how="inner")
    same_moment = (
        standard[["user_id", "time_ms"]]
        .drop_duplicates()
        .merge(random_log[["user_id", "time_ms"]].drop_duplicates(),
               on=["user_id", "time_ms"], how="inner")
    )

    report = {
        "shared_impressions": int(len(shared)),
        "same_user_and_millisecond": int(len(same_moment)),
    }
    if report["shared_impressions"]:
        raise ValueError(
            f"log_standard and log_random share {report['shared_impressions']} impressions; "
            "merging them would double-count. Investigate before continuing."
        )
    logger.info(
        "logs are disjoint (%d shared impressions); %d user-millisecond collisions, "
        "resolved by a deterministic sort tie-break",
        report["shared_impressions"],
        report["same_user_and_millisecond"],
    )
    return report


def exposure_summary(df: pd.DataFrame, actions: np.ndarray | None = None) -> dict:
    """Random vs recommended shares, and how differently users react to each."""
    is_rand = df["is_rand"].to_numpy() == 1
    n = int(is_rand.size)
    summary = {
        "n_interactions": n,
        "n_random_exposure": int(is_rand.sum()),
        "n_recommended": int((~is_rand).sum()),
        "random_share": float(is_rand.mean()) if n else 0.0,
    }

    # The same feedback rate under both policies is the headline bias number.
    for column in ("is_click", "long_view", "is_like"):
        if column not in df.columns:
            continue
        values = df[column].to_numpy()
        rec_rate = float(values[~is_rand].mean()) if (~is_rand).any() else 0.0
        rand_rate = float(values[is_rand].mean()) if is_rand.any() else 0.0
        summary[f"{column}_rate"] = {
            "recommended": rec_rate,
            "random": rand_rate,
            "ratio": float(rec_rate / rand_rate) if rand_rate else None,
        }

    if actions is not None:
        summary["action_distribution_recommended"] = (
            action_distribution(actions[~is_rand].astype(np.int64)) if (~is_rand).any() else {}
        )
        summary["action_distribution_random"] = (
            action_distribution(actions[is_rand].astype(np.int64)) if is_rand.any() else {}
        )

    return summary


def exposure_by_split(df: pd.DataFrame, split_codes: np.ndarray) -> dict:
    """Random-exposure share inside each split.

    Uneven by construction: random exposures only cover the second half of the
    window, so the later the split, the more unbiased data it holds.
    """
    is_rand = df["is_rand"].to_numpy() == 1
    out = {}
    for name, code in SPLIT_CODES.items():
        mask = split_codes == code
        if not mask.any():
            out[name] = None
            continue
        out[name] = {
            "n_interactions": int(mask.sum()),
            "n_random_exposure": int((mask & is_rand).sum()),
            "random_share": float(is_rand[mask].mean()),
        }
    return out
