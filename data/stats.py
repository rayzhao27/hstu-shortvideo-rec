"""Profile a KuaiRand log: the Stage 0 acceptance numbers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data import schema

SEQ_LEN_QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)


@dataclass
class LogStats:
    """Serialisable profile of one interaction log."""

    split: str
    n_interactions: int
    n_users: int
    n_items: int
    mean_seq_len: float
    median_seq_len: float
    min_seq_len: int
    max_seq_len: int
    seq_len_quantiles: dict
    density: float
    time_start: str
    time_end: str
    n_days: int
    interactions_per_day: float
    feedback_rates: dict
    label_consistency: dict
    popularity: dict
    duration_seconds: dict
    play_ratio: dict
    is_rand_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _describe(series: pd.Series, quantiles) -> dict:
    out = {"mean": float(series.mean()), "min": float(series.min()), "max": float(series.max())}
    for q in quantiles:
        out[f"p{q * 100:g}"] = float(series.quantile(q))
    return out


def _timestamp_of(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        return df["timestamp"]
    return pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert(schema.DATASET_TZ)


def compute_stats(df: pd.DataFrame, split: str = "standard") -> LogStats:
    """Compute the Stage 0 profile. Validates the schema first."""
    schema.validate_log(df)

    timestamp = _timestamp_of(df)
    seq_len = df.groupby("user_id").size()
    item_pop = df.groupby("video_id").size().sort_values(ascending=False)
    n_users, n_items, n_rows = seq_len.size, item_pop.size, len(df)

    start, end = timestamp.min(), timestamp.max()
    n_days = int((end - start).days) + 1

    # Clip at 5x: a handful of rows report replays far beyond the video length.
    play_ratio = (df["play_time_ms"] / df["duration_ms"].clip(lower=1)).clip(upper=5)

    feedback_rates = {
        column: float(df[column].mean())
        for column in schema.FEEDBACK_COLUMNS
        if column in df.columns
    }

    label_consistency = {
        "is_click_vs_valid_play": float(
            (
                df["is_click"].to_numpy()
                == schema.valid_play(df["play_time_ms"], df["duration_ms"])
            ).mean()
        ),
        "long_view_vs_rule": float(
            (
                df["long_view"].to_numpy()
                == schema.long_view(df["play_time_ms"], df["duration_ms"])
            ).mean()
        ),
    }

    popularity = {
        "top1pct_impression_share": float(item_pop.head(max(1, n_items // 100)).sum() / n_rows),
        "top10pct_impression_share": float(item_pop.head(max(1, n_items // 10)).sum() / n_rows),
        "items_below_10_impressions": float((item_pop < 10).mean()),
        "max_item_impressions": int(item_pop.iloc[0]),
    }

    return LogStats(
        split=split,
        n_interactions=n_rows,
        n_users=int(n_users),
        n_items=int(n_items),
        mean_seq_len=float(seq_len.mean()),
        median_seq_len=float(seq_len.median()),
        min_seq_len=int(seq_len.min()),
        max_seq_len=int(seq_len.max()),
        seq_len_quantiles={f"p{q * 100:g}": float(seq_len.quantile(q)) for q in SEQ_LEN_QUANTILES},
        density=float(n_rows / (n_users * n_items)),
        time_start=start.isoformat(),
        time_end=end.isoformat(),
        n_days=n_days,
        interactions_per_day=float(n_rows / n_days),
        feedback_rates=feedback_rates,
        label_consistency=label_consistency,
        popularity=popularity,
        duration_seconds=_describe(df["duration_ms"] / 1000, (0.25, 0.5, 0.75, 0.95)),
        play_ratio={
            **_describe(play_ratio, (0.25, 0.5, 0.75, 0.95)),
            "finished_share": float((play_ratio >= 1.0).mean()),
            "skipped_share": float((play_ratio < 0.2).mean()),
        },
        is_rand_counts={str(k): int(v) for k, v in df["is_rand"].value_counts().items()},
    )


def format_report(stats: LogStats) -> str:
    """Render LogStats as the human-readable Stage 0 report."""
    rule = "=" * 66
    lines = [
        rule,
        f" KuaiRand / {stats.split} log",
        rule,
        f" users                    : {stats.n_users:,}",
        f" items                    : {stats.n_items:,}",
        f" interactions             : {stats.n_interactions:,}",
        f" mean seq len per user    : {stats.mean_seq_len:.1f}",
        f" median / min / max       : {stats.median_seq_len:.0f} / {stats.min_seq_len:,}"
        f" / {stats.max_seq_len:,}",
        f" density                  : {stats.density:.4%}",
        f" time span                : {stats.time_start[:16]} -> {stats.time_end[:16]}"
        f"  ({stats.n_days} days)",
        f" interactions per day     : {stats.interactions_per_day:,.0f}",
        "-" * 66,
        " sequence length quantiles (drives max_seq_len):",
    ]
    lines += [f"   {k:<6}: {v:>8,.0f}" for k, v in stats.seq_len_quantiles.items()]
    lines += ["-" * 66, " feedback rates:"]
    lines += [f"   {k:<18}: {v:>9.4%}" for k, v in stats.feedback_rates.items()]
    lines += ["-" * 66, " label rules recomputed from play_time_ms / duration_ms:"]
    lines += [f"   {k:<26}: {v:>7.4f}" for k, v in stats.label_consistency.items()]
    lines += [
        "-" * 66,
        " item popularity:",
        f"   top  1% items hold      : {stats.popularity['top1pct_impression_share']:.2%}"
        " of impressions",
        f"   top 10% items hold      : {stats.popularity['top10pct_impression_share']:.2%}"
        " of impressions",
        f"   items with <10 impr.    : {stats.popularity['items_below_10_impressions']:.2%}",
        "-" * 66,
        " watch behaviour:",
        f"   duration median         : {stats.duration_seconds['p50']:.1f} s",
        f"   play ratio median       : {stats.play_ratio['p50']:.3f}",
        f"   finished (ratio >= 1.0) : {stats.play_ratio['finished_share']:.2%}",
        f"   skipped  (ratio <  0.2) : {stats.play_ratio['skipped_share']:.2%}",
        f"   is_rand counts          : {stats.is_rand_counts}",
        rule,
    ]
    return "\n".join(lines)


def write_stats(stats_by_split: dict, path: Path) -> Path:
    """Persist the profile as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {split: stats.to_dict() for split, stats in stats_by_split.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


@dataclass
class UserTimeline:
    """One user's chronological impression stream, for eyeballing the data."""

    user_id: int
    n_impressions: int
    n_sessions: int
    timeline: pd.DataFrame


def sample_user_timeline(
    df: pd.DataFrame, n_rows: int = 30, user_id: int | None = None
) -> UserTimeline:
    """Sample a user and return their first n_rows impressions in time order.

    Defaults to the user whose sequence length is closest to the median, so the
    sample is neither a power user nor a single-impression account.
    """
    if user_id is None:
        seq_len = df.groupby("user_id").size()
        user_id = int((seq_len - seq_len.median()).abs().idxmin())

    rows = df[df["user_id"] == user_id].sort_values("time_ms")
    timestamp = _timestamp_of(rows)

    timeline = pd.DataFrame(
        {
            "time": timestamp.dt.strftime("%m-%d %H:%M:%S"),
            "gap_s": rows["time_ms"].diff().div(1000).fillna(0).round(1),
            "video_id": rows["video_id"],
            "duration_s": (rows["duration_ms"] / 1000).round(1),
            "play_s": (rows["play_time_ms"] / 1000).round(1),
            "ratio": (rows["play_time_ms"] / rows["duration_ms"].clip(lower=1)).round(2),
            "click": rows["is_click"],
            "long_view": rows["long_view"],
            "like": rows["is_like"],
        }
    ).reset_index(drop=True)

    # Impressions more than 30 min apart belong to different browsing sessions.
    gaps = rows["time_ms"].diff().fillna(np.inf)
    return UserTimeline(
        user_id=user_id,
        n_impressions=len(rows),
        n_sessions=int((gaps > 30 * 60 * 1000).sum()),
        timeline=timeline.head(n_rows),
    )
