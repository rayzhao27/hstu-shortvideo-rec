"""Overview figures for a KuaiRand log, saved under pictures/."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data.schema import DATASET_TZ  # noqa: E402

logger = logging.getLogger(__name__)

PICTURES_DIR = Path("pictures")


def save_log_overview(
    df: pd.DataFrame, out_dir: Path = PICTURES_DIR, split: str = "standard"
) -> Path:
    """Write a four-panel overview PNG and return its path.

    Labels are English on purpose: the default matplotlib font ships no CJK
    glyphs, so Chinese titles render as empty boxes in Colab.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_len = df.groupby("user_id").size()
    item_pop = df.groupby("video_id").size().sort_values(ascending=False)
    if "timestamp" in df.columns:
        timestamp = df["timestamp"]
    else:
        timestamp = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert(DATASET_TZ)

    fig, axes = plt.subplots(1, 4, figsize=(20, 3.6))

    axes[0].hist(seq_len, bins=60, color="#4C78A8")
    axes[0].set(
        title="Sequence length per user",
        xlabel="interactions / user",
        ylabel="users (log)",
        yscale="log",
    )

    axes[1].plot(np.arange(1, item_pop.size + 1), item_pop.to_numpy(), color="#E45756")
    axes[1].set(
        title="Item popularity (long tail)",
        xlabel="video rank",
        ylabel="impressions",
        xscale="log",
        yscale="log",
    )

    axes[2].hist((df["duration_ms"] / 1000).clip(upper=120), bins=60, color="#54A24B")
    axes[2].set(
        title="Video duration (s, clipped at 120)", xlabel="duration (s)", ylabel="impressions"
    )

    hourly = timestamp.dt.hour.value_counts().sort_index()
    axes[3].bar(hourly.index, hourly.to_numpy(), color="#B279A2")
    axes[3].set(
        title=f"Activity by hour ({DATASET_TZ})", xlabel="hour of day", ylabel="impressions"
    )

    fig.tight_layout()
    path = out_dir / f"stage0_overview_{split}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def save_action_distribution(stats: dict, out_dir: Path = PICTURES_DIR) -> Path:
    """Action distribution overall, and split by exposure policy.

    The right panel is the point of the figure: the same action taxonomy under
    recommended versus random exposure shows how much of the observed engagement is
    the old policy's doing rather than the user's preference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = stats["action_distribution"]
    names = list(overall)
    counts = [overall[n]["count"] for n in names]

    exp = stats["exposure"]
    rec = exp.get("action_distribution_recommended") or {}
    rnd = exp.get("action_distribution_random") or {}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

    axes[0].bar(names, counts, color="#4C78A8")
    axes[0].set(title="Action distribution (after filtering)", ylabel="interactions",
                yscale="log")
    axes[0].tick_params(axis="x", rotation=30)
    for x, count in enumerate(counts):
        axes[0].text(x, count, f"{overall[names[x]]['share']:.2%}", ha="center",
                     va="bottom", fontsize=8)

    if rec and rnd:
        width = 0.38
        positions = np.arange(len(names))
        axes[1].bar(positions - width / 2, [rec[n]["share"] for n in names], width,
                    label="recommended (is_rand=0)", color="#4C78A8")
        axes[1].bar(positions + width / 2, [rnd[n]["share"] for n in names], width,
                    label="random (is_rand=1)", color="#E45756")
        axes[1].set(title="Action share by exposure policy", ylabel="share of impressions",
                    yscale="log")
        axes[1].set_xticks(positions, names, rotation=30)
        axes[1].legend(fontsize=8)

        ratios = [
            rec[n]["share"] / rnd[n]["share"] if rnd[n]["share"] else np.nan for n in names
        ]
        axes[2].axhline(1.0, color="#888888", linestyle="--", linewidth=1)
        axes[2].bar(names, ratios, color="#54A24B")
        axes[2].set(title="Exposure bias: recommended / random", ylabel="rate ratio")
        axes[2].tick_params(axis="x", rotation=30)
        for x, ratio in enumerate(ratios):
            if np.isfinite(ratio):
                axes[2].text(x, ratio, f"{ratio:.1f}x", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    path = out_dir / "stage1_action_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def save_pipeline_overview(stats: dict, df: pd.DataFrame, out_dir: Path = PICTURES_DIR) -> Path:
    """Filtering funnel, sequence lengths after filtering, and split composition."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.0))

    # Funnel: share kept at each stage of the cleaning pipeline.
    dedupe_stats, filt = stats["dedupe"], stats["filtering"]
    stages = ["raw", "deduped", "filtered"]
    values = [dedupe_stats["n_before"], dedupe_stats["n_after"], filt["after"]["n_interactions"]]
    axes[0].bar(stages, values, color=["#B0B0B0", "#4C78A8", "#54A24B"])
    axes[0].set(title="Interactions kept per stage", ylabel="interactions")
    for x, value in enumerate(values):
        axes[0].text(x, value, f"{value / values[0]:.1%}", ha="center", va="bottom", fontsize=9)

    # Users and items before / after k-core filtering.
    width = 0.38
    positions = np.arange(2)
    axes[1].bar(positions - width / 2, [filt["before"]["n_users"], filt["before"]["n_items"]],
                width, label="before", color="#B0B0B0")
    axes[1].bar(positions + width / 2, [filt["after"]["n_users"], filt["after"]["n_items"]],
                width, label="after", color="#4C78A8")
    axes[1].set(title=f"k-core filtering ({filt['n_rounds']} rounds)", yscale="log")
    axes[1].set_xticks(positions, ["users", "items"])
    axes[1].legend(fontsize=8)

    seq_len = df.groupby("user_idx").size()
    axes[2].hist(seq_len, bins=60, color="#F58518")
    axes[2].set(title="Sequence length after filtering", xlabel="interactions / user",
                ylabel="users (log)", yscale="log")

    # Split composition, separating biased from unbiased interactions.
    by_split = stats["exposure"]["by_split"]
    names = [n for n, v in by_split.items() if v]
    random_counts = [by_split[n]["n_random_exposure"] for n in names]
    rec_counts = [by_split[n]["n_interactions"] - by_split[n]["n_random_exposure"] for n in names]
    axes[3].bar(names, rec_counts, label="recommended", color="#4C78A8")
    axes[3].bar(names, random_counts, bottom=rec_counts, label="random", color="#E45756")
    axes[3].set(title="Split composition by exposure", ylabel="interactions")
    axes[3].legend(fontsize=8)
    for x, name in enumerate(names):
        axes[3].text(x, by_split[name]["n_interactions"],
                     f"{by_split[name]['random_share']:.0%} rand", ha="center", va="bottom",
                     fontsize=8)

    fig.tight_layout()
    path = out_dir / "stage1_pipeline.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def save_stage1_plots(stats: dict, df: pd.DataFrame, out_dir: Path = PICTURES_DIR) -> list[Path]:
    return [
        save_action_distribution(stats, out_dir),
        save_pipeline_overview(stats, df, out_dir),
    ]
