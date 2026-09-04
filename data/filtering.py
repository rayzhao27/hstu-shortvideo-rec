"""Iterative k-core filtering of the interaction table.

Dropping short user sequences makes some items fall below the item threshold, and
dropping cold items shortens more user sequences. A single pass therefore leaves
users and items that violate the thresholds, which is why this iterates until the
table stops shrinking. Every round is recorded so the funnel can be reported.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterRound:
    """State of the table after one filtering round."""

    iteration: int
    n_interactions: int
    n_users: int
    n_items: int
    dropped_interactions: int
    dropped_users: int
    dropped_items: int

    def to_dict(self) -> dict:
        return asdict(self)


def _snapshot(df: pd.DataFrame, iteration: int, previous: FilterRound | None) -> FilterRound:
    n_interactions = len(df)
    n_users = df["user_id"].nunique()
    n_items = df["video_id"].nunique()
    if previous is None:
        return FilterRound(iteration, n_interactions, n_users, n_items, 0, 0, 0)
    return FilterRound(
        iteration=iteration,
        n_interactions=n_interactions,
        n_users=n_users,
        n_items=n_items,
        dropped_interactions=previous.n_interactions - n_interactions,
        dropped_users=previous.n_users - n_users,
        dropped_items=previous.n_items - n_items,
    )


def kcore_filter(
    df: pd.DataFrame,
    min_user_len: int = 20,
    min_item_count: int = 10,
    max_iters: int = 20,
) -> tuple[pd.DataFrame, list[FilterRound]]:
    """Drop users with too few interactions and items with too few impressions.

    Returns the filtered table and the per-round funnel, including round 0 which is
    the state before any filtering.
    """
    rounds = [_snapshot(df, 0, None)]
    logger.info(
        "filtering: start with %d interactions, %d users, %d items "
        "(min_user_len=%d, min_item_count=%d)",
        rounds[0].n_interactions,
        rounds[0].n_users,
        rounds[0].n_items,
        min_user_len,
        min_item_count,
    )

    for iteration in range(1, max_iters + 1):
        user_counts = df["user_id"].map(df["user_id"].value_counts())
        item_counts = df["video_id"].map(df["video_id"].value_counts())
        keep = (user_counts >= min_user_len) & (item_counts >= min_item_count)

        if keep.all():
            logger.info("filtering: converged after %d round(s)", iteration - 1)
            break

        df = df[keep]
        if df.empty:
            raise ValueError(
                f"filtering removed every interaction; min_user_len={min_user_len} and "
                f"min_item_count={min_item_count} are too aggressive for this dataset"
            )

        current = _snapshot(df, iteration, rounds[-1])
        rounds.append(current)
        logger.info(
            "  round %d: -%d interactions, -%d users, -%d items -> %d / %d / %d",
            iteration,
            current.dropped_interactions,
            current.dropped_users,
            current.dropped_items,
            current.n_interactions,
            current.n_users,
            current.n_items,
        )
    else:
        logger.warning("filtering did not converge in %d rounds", max_iters)

    return df.reset_index(drop=True), rounds


def filtering_summary(rounds: Iterable[FilterRound]) -> dict:
    """Before / after view of the funnel, for stats.json."""
    rounds = list(rounds)
    first, last = rounds[0], rounds[-1]
    return {
        "before": {
            "n_interactions": first.n_interactions,
            "n_users": first.n_users,
            "n_items": first.n_items,
        },
        "after": {
            "n_interactions": last.n_interactions,
            "n_users": last.n_users,
            "n_items": last.n_items,
        },
        "kept_share": {
            "interactions": last.n_interactions / first.n_interactions,
            "users": last.n_users / first.n_users,
            "items": last.n_items / first.n_items,
        },
        "n_rounds": len(rounds) - 1,
        "rounds": [r.to_dict() for r in rounds],
    }
