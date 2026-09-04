"""Turn the flat interaction table into per-user sequences, one file per split.

Layout of a split file: a list of records, one per user, each holding the user's
chronological history *up to the end of that split's window* plus a boolean mask
marking which positions the split scores::

    {
        "user": 731,                        # remapped user index
        "items":      np.int32   [n],       # remapped video indices, chronological
        "actions":    np.int8    [n],       # action ids from data.actions
        "timestamps": np.int64   [n],       # time_ms
        "is_rand":    np.int8    [n],       # 1 = random exposure
        "is_target":  np.bool_   [n],       # positions this split predicts
        "n_targets":  17,                   # is_target.sum(), for convenience
    }

Why store the sequence once with a target mask, instead of materialising
(prefix, target) pairs: the model conditions on everything before a target anyway, so
this is both far smaller and impossible to get wrong. Leakage is structural rather
than enforced by a check - the val file simply contains no test interactions, and the
train file contains neither.

Why a mask instead of "the last k positions": once targets can be restricted to a
subset of impressions (see ``--target-policy``), they are no longer a contiguous
suffix. A mask covers both cases with one format and removes an invariant that a
later refactor could silently break.

The mask always excludes position 0, since a target is scored from the prefix before
it and position 0 has none.

One consequence is deliberate: masking positions rather than dropping users makes
"restrict targets during preprocessing" and "restrict targets when reading the file"
produce exactly the same target set. That is what lets a single set of artifacts serve
both evaluation streams, and lets every model be compared on identical targets.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data.actions import action_distribution
from data.splitting import SPLIT_CODES


logger = logging.getLogger(__name__)


def _user_slices(user_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end offsets of each user block. Requires user-sorted input."""
    if user_idx.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, user_idx[1:] != user_idx[:-1]])
    ends = np.r_[starts[1:], user_idx.size]
    return starts, ends


def _assert_codes_monotonic(user_idx: np.ndarray, codes: np.ndarray) -> None:
    """Split codes must never decrease inside a user block.

    Both split strategies put the evaluation windows at the end of a user's timeline.
    A decrease means the table is not sorted by (user_id, time_ms), which would put
    later interactions into an earlier split's history: exactly the leakage this
    module is supposed to make impossible.
    """
    if user_idx.size < 2:
        return
    same_user = user_idx[1:] == user_idx[:-1]
    decreasing = np.diff(codes.astype(np.int16)) < 0
    if bool((same_user & decreasing).any()):
        raise ValueError(
            "split codes decrease within a user block: the table is not sorted by "
            "(user_id, time_ms), or the split strategy does not put targets last"
        )


def build_split_sequences(
    df: pd.DataFrame,
    split_codes: np.ndarray,
    split: str,
    target_mask: np.ndarray | None = None,
    min_history: int = 1,
) -> tuple[list[dict], dict]:
    """Build the sequence records for one split.

    ``df`` must be sorted by (user_idx, time_ms) and already carry remapped ids.
    ``target_mask`` optionally narrows which interactions may be scored; history is
    never narrowed, so the model always sees the user's complete past.

    The first ``min_history`` positions are never marked as targets, since a position
    is scored from the prefix before it. A user is dropped only when nothing is left to
    score. Both counts are returned in the stats.
    """
    code = SPLIT_CODES[split]
    visible = split_codes <= code

    sub = df.loc[visible]
    sub_codes = split_codes[visible]
    user_idx = sub["user_idx"].to_numpy()
    _assert_codes_monotonic(user_idx, sub_codes)

    items = sub["item_idx"].to_numpy(dtype=np.int32)
    actions = sub["action"].to_numpy(dtype=np.int8)
    times = sub["time_ms"].to_numpy(dtype=np.int64)
    is_rand = sub["is_rand"].to_numpy(dtype=np.int8)

    eligible = sub_codes == code
    if target_mask is not None:
        eligible = eligible & target_mask[visible]

    # A position is scored from the prefix that precedes it, so the first `horizon`
    # positions can never be targets. At least one, because position 0 has no prefix.
    horizon = max(min_history, 1)

    records: list[dict] = []
    dropped_no_target = 0
    n_targets_without_history = 0
    target_actions: list[np.ndarray] = []
    target_is_rand: list[np.ndarray] = []

    starts, ends = _user_slices(user_idx)
    for start, end in zip(starts, ends):
        is_target = eligible[start:end].copy()

        # Un-mark the un-conditionable head rather than dropping the user: a user whose
        # first target lacks history usually has plenty of later ones that do, and
        # dropping them would also make the target set depend on which exposure stream
        # happens to come first, breaking the equivalence that lets one artifact serve
        # both evaluation streams.
        n_targets_without_history += int(is_target[:horizon].sum())
        is_target[:horizon] = False

        n_targets = int(is_target.sum())
        if n_targets == 0:
            dropped_no_target += 1
            continue

        records.append(
            {
                "user": int(user_idx[start]),
                "items": items[start:end].copy(),
                "actions": actions[start:end].copy(),
                "timestamps": times[start:end].copy(),
                "is_rand": is_rand[start:end].copy(),
                "is_target": is_target,
                "n_targets": n_targets,
            }
        )
        target_actions.append(actions[start:end][is_target])
        target_is_rand.append(is_rand[start:end][is_target])

    stats = _summarise(
        split=split,
        records=records,
        dropped_no_target=dropped_no_target,
        n_targets_without_history=n_targets_without_history,
        target_actions=np.concatenate(target_actions) if target_actions else np.empty(0, np.int8),
        target_is_rand=np.concatenate(target_is_rand) if target_is_rand else np.empty(0, np.int8),
    )
    logger.info(
        "%-5s: %d users, %d targets, seq len mean %.1f (dropped %d users without "
        "targets, unmarked %d targets without history)",
        split,
        stats["n_users"],
        stats["n_targets"],
        stats["seq_len"]["mean"],
        dropped_no_target,
        n_targets_without_history,
    )
    return records, stats


def _summarise(
    split: str,
    records: list[dict],
    dropped_no_target: int,
    n_targets_without_history: int,
    target_actions: np.ndarray,
    target_is_rand: np.ndarray,
) -> dict:
    seq_lens = np.array([len(r["items"]) for r in records], dtype=np.int64)
    target_counts = np.array([r["n_targets"] for r in records], dtype=np.int64)

    def _len_stats(values: np.ndarray) -> dict:
        if values.size == 0:
            return {"mean": 0.0, "median": 0.0, "min": 0, "max": 0, "p95": 0.0}
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": int(values.min()),
            "max": int(values.max()),
            "p95": float(np.quantile(values, 0.95)),
        }

    return {
        "split": split,
        "n_users": len(records),
        "n_interactions": int(seq_lens.sum()),
        "n_targets": int(target_counts.sum()),
        "seq_len": _len_stats(seq_lens),
        "targets_per_user": _len_stats(target_counts),
        "dropped_users_without_targets": dropped_no_target,
        "n_targets_unmarked_without_history": n_targets_without_history,
        "targets_by_stream": {
            "recommended": int(sum(int((r["is_target"] & (r["is_rand"] == 0)).sum())
                                   for r in records)),
            "random": int(sum(int((r["is_target"] & (r["is_rand"] == 1)).sum())
                              for r in records)),
        },
        "users_by_stream": {
            "recommended": int(sum(bool((r["is_target"] & (r["is_rand"] == 0)).any())
                                   for r in records)),
            "random": int(sum(bool((r["is_target"] & (r["is_rand"] == 1)).any())
                              for r in records)),
        },
        "target_action_distribution": (
            action_distribution(target_actions.astype(np.int64)) if target_actions.size else {}
        ),
        "target_random_exposure_share": (
            float(target_is_rand.mean()) if target_is_rand.size else 0.0
        ),
    }


def save_sequences(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "wrote %s (%d users, %.1f MB)", path, len(records), path.stat().st_size / 1024**2
    )
    return path


def load_sequences(path: Path) -> list[dict]:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def verify_no_future_leakage(sequences_by_split: dict) -> dict:
    """Read the records back and check no split holds data past a later split's targets.

    The file format makes this true by construction, but "by construction" is exactly
    the kind of claim that stops holding after a refactor, so it is asserted on the
    actual arrays.
    """
    max_time: dict = {}
    min_target_time: dict = {}

    for split, records in sequences_by_split.items():
        if not records:
            continue
        max_time[split] = max(int(r["timestamps"].max()) for r in records)
        min_target_time[split] = min(
            int(r["timestamps"][r["is_target"]].min()) for r in records
        )

    report = {
        "max_time_ms": max_time,
        "min_target_time_ms": min_target_time,
        "violations": [],
    }
    for earlier, later in (("train", "val"), ("val", "test"), ("train", "test")):
        if earlier in max_time and later in min_target_time:
            if max_time[earlier] > min_target_time[later]:
                report["violations"].append(
                    f"{earlier} history extends past the first {later} target"
                )
    report["leak_free"] = not report["violations"]
    return report
