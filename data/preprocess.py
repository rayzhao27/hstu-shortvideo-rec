"""Stage 1: build HSTU-ready user sequences from the raw KuaiRand logs.

Run from the repo root:

    python -m data.preprocess                            # defaults below
    python -m data.preprocess --split-strategy loo
    python -m data.preprocess --min-user-len 30 --min-item-count 20
    python -m data.preprocess --no-random                 # biased log only

Pipeline, in order:

  1. load log_standard (+ log_random) and check the two logs are disjoint
  2. drop duplicate impressions on (user_id, video_id, time_ms)
  3. sort into per-user chronological order with a deterministic tie-break
  4. encode each impression as one action id (data.actions)
  5. iterative k-core filtering of short users and cold items (data.filtering)
  6. assign train / val / test by time (data.splitting) and measure leakage
  7. fit id encoders on the train split, remap user_id / video_id (data.encoders)
  8. write one sequence file per split (data.sequences)

Outputs under datasets/processed/:
    train_seqs.pkl, val_seqs.pkl, test_seqs.pkl   per-user sequences + target suffix
    item_encoder.pkl, user_encoder.pkl            id mappings
    interactions.parquet                          flat encoded table, for debugging
    preprocess_stats.json                         every number reported below
and pictures/stage1_*.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import actions as actions_mod
from data import exposure, filtering, loader, sequences, splitting
from data.download import DATASETS, RAW_DIR, ensure_dataset
from data.encoders import IdEncoder
from data.loader import PROCESSED_DIR

logger = logging.getLogger("data.preprocess")

# Deterministic order. is_rand and video_id break ties when two impressions share a
# millisecond (about 11k cases), so the sequence order never depends on file order.
SORT_KEYS = ["user_id", "time_ms", "is_rand", "video_id"]

DEDUPE_KEYS = ["user_id", "video_id", "time_ms"]

KEEP_COLUMNS = [
    "user_id",
    "video_id",
    "time_ms",
    "is_rand",
    "is_click",
    "long_view",
    "is_like",
    "is_follow",
    "is_comment",
    "is_forward",
    "is_hate",
    "is_profile_enter",
    "play_time_ms",
    "duration_ms",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: sequence construction.")
    parser.add_argument("--dataset", default="KuaiRand-Pure", choices=sorted(DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--pictures-dir", type=Path, default=Path("pictures"))

    parser.add_argument("--no-random", dest="include_random", action="store_false",
                        help="use only the biased standard log")
    parser.add_argument("--min-user-len", type=int, default=20,
                        help="drop users with fewer interactions (default: 20)")
    parser.add_argument("--min-item-count", type=int, default=10,
                        help="drop items with fewer impressions (default: 10)")
    parser.add_argument("--split-strategy", default="temporal", choices=["temporal", "loo"])
    parser.add_argument("--val-days", type=float, default=3.0, help="temporal split only")
    parser.add_argument("--test-days", type=float, default=4.0, help="temporal split only")
    parser.add_argument("--min-history", type=int, default=1,
                        help="leading positions that may not be scored, because a "
                             "target is predicted from the prefix before it")
    parser.add_argument("--target-policy", default="all",
                        choices=["all", "recommended", "random"],
                        help="which impressions may be prediction targets; history "
                             "always keeps every impression. Leave at 'all' for the "
                             "agreed protocol and select the stream at read time with "
                             "data.protocol (default: all)")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_raw(data_dir: Path, include_random: bool) -> tuple[pd.DataFrame, dict]:
    """Load the logs, verify they are disjoint, and concatenate them."""
    standard = loader.load_log(data_dir, "standard", with_timestamp=False, sort=False)
    report = {"n_standard": len(standard), "n_random": 0, "disjoint_check": None}

    if not include_random:
        logger.warning("--no-random: dropping the unbiased random-exposure log")
        return standard[KEEP_COLUMNS], report

    random_log = loader.load_log(data_dir, "random", with_timestamp=False, sort=False)
    report["n_random"] = len(random_log)
    report["disjoint_check"] = exposure.assert_logs_disjoint(standard, random_log)

    merged = pd.concat([standard[KEEP_COLUMNS], random_log[KEEP_COLUMNS]], ignore_index=True)
    return merged, report


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop repeated impressions of the same video by the same user at the same ms.

    1.63% of the standard log is duplicated this way, and 94% of those duplicates are
    byte-identical rows, so this is logging noise rather than two real impressions.
    Where the duplicates disagree on labels (18 groups) the row with the longest play
    time wins: it is the most complete observation of that impression.

    Note this is *not* deduplication of (user, video). Once the logging noise is gone,
    2.5% of the standard log's user-video pairs are still shown more than once at
    different times, up to 22 times, and those re-exposures are real signal for a
    sequence model.
    """
    before = len(df)
    ordered = df.sort_values("play_time_ms", ascending=False, kind="mergesort")
    deduped = ordered.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    removed = before - len(deduped)

    repeats = deduped.groupby(["user_id", "video_id"]).size()
    stats = {
        "n_before": before,
        "n_after": int(len(deduped)),
        "n_removed": int(removed),
        "removed_share": float(removed / before) if before else 0.0,
        "repeated_user_item_pairs": int((repeats > 1).sum()),
        "repeated_user_item_share": float((repeats > 1).mean()),
        "max_repeats_of_one_pair": int(repeats.max()),
    }
    logger.info(
        "dedupe: removed %d duplicate impressions (%.3f%%); kept %d user-video pairs "
        "that legitimately repeat (max %d times)",
        removed,
        stats["removed_share"] * 100,
        stats["repeated_user_item_pairs"],
        stats["max_repeats_of_one_pair"],
    )
    return deduped, stats


def fit_encoders(df: pd.DataFrame, split_codes: np.ndarray) -> tuple[IdEncoder, IdEncoder, dict]:
    """Fit the item encoder on train rows only, the user encoder on every kept user."""
    train_mask = split_codes == splitting.TRAIN
    item_encoder = IdEncoder.fit(df.loc[train_mask, "video_id"].to_numpy(), "video_id")
    user_encoder = IdEncoder.fit(df["user_id"].to_numpy(), "user_id")

    oov = {}
    for name, code in splitting.SPLIT_CODES.items():
        mask = split_codes == code
        if not mask.any():
            oov[name] = None
            continue
        values = df.loc[mask, "video_id"].to_numpy()
        is_oov = item_encoder.oov_mask(values)
        oov[name] = {
            "n_interactions": int(mask.sum()),
            "n_oov_interactions": int(is_oov.sum()),
            "oov_share": float(is_oov.mean()),
            "n_oov_items": int(pd.unique(values[is_oov]).size),
        }

    # Train users are a subset of all kept users, so the difference is the count of
    # users whose first interaction lands in the val or test window.
    users_without_train = int(df["user_id"].nunique() - df.loc[train_mask, "user_id"].nunique())

    summary = {
        "item_encoder": item_encoder.summary(),
        "user_encoder": user_encoder.summary(),
        "item_oov_by_split": oov,
        "users_absent_from_train": users_without_train,
    }
    logger.info(
        "encoders: %d items (train-only vocabulary), %d users; %d users have no train history",
        item_encoder.n_known,
        user_encoder.n_known,
        users_without_train,
    )
    return item_encoder, user_encoder, summary


def assert_chronological(df: pd.DataFrame, user_column: str) -> None:
    """Everything downstream assumes user-major, time-ascending order.

    The table is sorted once after deduplication; filtering uses a boolean mask and
    remapping adds columns, so neither can reorder it. This asserts that rather than
    re-sorting defensively, because a redundant sort would hide a reordering bug
    instead of surfacing it.
    """
    users = df[user_column].to_numpy()
    times = df["time_ms"].to_numpy()
    if users.size < 2:
        return
    if bool((users[1:] < users[:-1]).any()):
        raise AssertionError(f"table is not sorted by {user_column}")
    if bool(((users[1:] == users[:-1]) & (np.diff(times) < 0)).any()):
        raise AssertionError(f"table is not sorted by time within a {user_column} block")


def build_target_mask(df: pd.DataFrame, policy: str) -> np.ndarray | None:
    """Restrict which impressions may be scored, without touching the history.

    KuaiRand-Pure makes this knob necessary rather than decorative: the standard log
    thins out from ~278k impressions/day in mid-April to ~14k/day by month end, while
    the random log runs at 40-110k/day from 04-22 onwards. A plain temporal split
    therefore trains on ~26% random exposures and tests on ~88% of them, so a naive
    train/test comparison measures the policy shift as much as the model.

    ``all``          score everything (default, most data)
    ``recommended``  score only is_rand=0, i.e. the production policy's own slate
    ``random``       score only is_rand=1, i.e. fully unbiased targets
    """
    if policy == "all":
        return None
    is_rand = df["is_rand"].to_numpy() == 1
    mask = ~is_rand if policy == "recommended" else is_rand
    logger.info(
        "target policy %r: %d of %d impressions are eligible targets (%.1f%%)",
        policy, int(mask.sum()), len(mask), 100 * mask.mean(),
    )
    return mask


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    pd.set_option("display.width", 200)

    data_dir = ensure_dataset(DATASETS[args.dataset])
    stats: dict = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    }

    # 1-2. load and clean
    df, load_report = load_raw(data_dir, args.include_random)
    stats["load"] = load_report
    df, stats["dedupe"] = dedupe(df)

    # 3. chronological order per user; the only sort in the pipeline
    df = df.sort_values(SORT_KEYS, kind="mergesort").reset_index(drop=True)
    assert_chronological(df, "user_id")

    # 4. action encoding
    print("\n" + actions_mod.describe_taxonomy() + "\n")
    df["action"] = actions_mod.encode_actions(df)
    stats["action_distribution_before_filtering"] = actions_mod.action_distribution(
        df["action"].to_numpy().astype(np.int64)
    )

    # 5. filtering
    df, rounds = filtering.kcore_filter(
        df, min_user_len=args.min_user_len, min_item_count=args.min_item_count
    )
    stats["filtering"] = filtering.filtering_summary(rounds)
    stats["action_distribution"] = actions_mod.action_distribution(
        df["action"].to_numpy().astype(np.int64)
    )

    assert_chronological(df, "user_id")

    # 6. split and leakage audit
    split_kwargs = (
        {"val_days": args.val_days, "test_days": args.test_days}
        if args.split_strategy == "temporal"
        else {}
    )
    split = splitting.make_split(df, args.split_strategy, **split_kwargs)
    stats["split"] = {
        "strategy": split.strategy,
        "boundaries": split.boundaries,
        "interactions": split.counts(),
        "leakage": splitting.leakage_report(df, split),
    }

    # 3. exposure bias
    stats["exposure"] = exposure.exposure_summary(df, df["action"].to_numpy())
    stats["exposure"]["by_split"] = exposure.exposure_by_split(df, split.codes)

    # 7. id remapping
    item_encoder, user_encoder, stats["encoders"] = fit_encoders(df, split.codes)
    df["item_idx"] = item_encoder.transform(df["video_id"].to_numpy())
    df["user_idx"] = user_encoder.transform(df["user_id"].to_numpy())
    df["split"] = split.codes
    # IdEncoder assigns indices in ascending raw-id order, so user_idx preserves the
    # user_id ordering and the table is still chronological. Asserted, not assumed.
    assert_chronological(df, "user_idx")
    split_codes = split.codes

    # 8. sequences per split
    target_mask = build_target_mask(df, args.target_policy)
    stats["target_policy"] = {
        "policy": args.target_policy,
        "n_eligible_targets": int(target_mask.sum()) if target_mask is not None else len(df),
    }

    stats["sequences"] = {}
    written = {}
    for split_name in splitting.SPLIT_NAMES:
        records, split_stats = sequences.build_split_sequences(
            df, split_codes, split_name,
            target_mask=target_mask,
            min_history=args.min_history,
        )
        stats["sequences"][split_name] = split_stats
        sequences.save_sequences(records, args.processed_dir / f"{split_name}_seqs.pkl")
        written[split_name] = records

    stats["leakage_verification"] = sequences.verify_no_future_leakage(written)
    if not stats["leakage_verification"]["leak_free"]:
        if split.strategy == "temporal":
            raise RuntimeError(
                f"temporal split leaked: {stats['leakage_verification']['violations']}"
            )
        logger.warning(
            "leave-last-one-out overlaps in time across users (expected): %s",
            stats["leakage_verification"]["violations"],
        )

    item_encoder.save(args.processed_dir / "item_encoder.pkl")
    user_encoder.save(args.processed_dir / "user_encoder.pkl")

    flat_columns = ["user_idx", "item_idx", "action", "time_ms", "is_rand", "split",
                    "user_id", "video_id"]
    loader.write_parquet(df[flat_columns], args.processed_dir / "interactions.parquet")

    stats_path = args.processed_dir / "preprocess_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info("wrote %s", stats_path)

    if args.plots:
        from visualization.save_plots import save_stage1_plots

        save_stage1_plots(stats, df, args.pictures_dir)

    print("\n" + format_summary(stats))
    return 0


def format_summary(stats: dict) -> str:
    """The Stage 1 acceptance report."""
    rule = "=" * 74
    lines = [rule, " STAGE 1 SUMMARY", rule]

    dedupe_stats = stats["dedupe"]
    lines += [
        f" raw impressions          : {dedupe_stats['n_before']:,}"
        f"  (standard {stats['load']['n_standard']:,} + random {stats['load']['n_random']:,})",
        f" after dedupe             : {dedupe_stats['n_after']:,}"
        f"  (-{dedupe_stats['n_removed']:,}, {dedupe_stats['removed_share']:.3%})",
    ]

    before, after = stats["filtering"]["before"], stats["filtering"]["after"]
    kept = stats["filtering"]["kept_share"]
    lines += [
        "-" * 74,
        f" filtering ({stats['filtering']['n_rounds']} rounds)     :"
        f" interactions {before['n_interactions']:,} -> {after['n_interactions']:,}"
        f" ({kept['interactions']:.1%})",
        f"                            users {before['n_users']:,} -> {after['n_users']:,}"
        f" ({kept['users']:.1%})",
        f"                            items {before['n_items']:,} -> {after['n_items']:,}"
        f" ({kept['items']:.1%})",
        "-" * 74,
        " action distribution (after filtering):",
    ]
    for name, entry in stats["action_distribution"].items():
        lines.append(f"   {entry['id']} {name:<10}: {entry['count']:>10,}  {entry['share']:>8.4%}")

    exp = stats["exposure"]
    lines += [
        "-" * 74,
        f" exposure bias            : random {exp['n_random_exposure']:,}"
        f" ({exp['random_share']:.2%}) vs recommended {exp['n_recommended']:,}",
    ]
    for column in ("is_click", "long_view", "is_like"):
        entry = exp.get(f"{column}_rate")
        if entry and entry["ratio"]:
            lines.append(
                f"   {column:<10} recommended {entry['recommended']:.4%}"
                f"  random {entry['random']:.4%}  x{entry['ratio']:.2f}"
            )
    lines.append("   random share per split: " + ", ".join(
        f"{name}={entry['random_share']:.1%}" if entry else f"{name}=-"
        for name, entry in exp["by_split"].items()
    ))

    train_share = (exp["by_split"].get("train") or {}).get("random_share")
    test_share = (exp["by_split"].get("test") or {}).get("random_share")
    if train_share is not None and test_share is not None and abs(test_share - train_share) > 0.2:
        lines += [
            f"   NOTE: the random share jumps from {train_share:.0%} in train to"
            f" {test_share:.0%} in test, because the",
            "   standard log thins out over the window while the random log ramps up."
            " Compare like",
            "   with like using --target-policy recommended|random, or slice on is_rand"
            " at eval time.",
        ]

    lines += [
        "-" * 74,
        f" split strategy           : {stats['split']['strategy']}"
        f"   target policy: {stats['target_policy']['policy']}",
    ]
    if stats["split"]["strategy"] == "temporal":
        lines.append(
            f"   train < {stats['split']['boundaries']['val_start'][:16]}"
            f" <= val < {stats['split']['boundaries']['test_start'][:16]} <= test"
        )
    for name, seq in stats["sequences"].items():
        lines.append(
            f"   {name:<5}: {seq['n_users']:>7,} users  {seq['n_targets']:>9,} targets"
            f"  seq len mean {seq['seq_len']['mean']:>6.1f}"
            f"  unbiased targets {seq['target_random_exposure_share']:.1%}"
        )
    leak = stats["leakage_verification"]
    lines.append(f"   leakage check          : {'PASS' if leak['leak_free'] else 'OVERLAP'}"
                 f" {'' if leak['leak_free'] else leak['violations']}")

    enc = stats["encoders"]
    lines += [
        "-" * 74,
        f" item vocabulary          : {enc['item_encoder']['n_known_ids']:,} items"
        f" (+PAD +UNK = {enc['item_encoder']['vocab_size']:,} embedding rows), fit on train",
        f" user vocabulary          : {enc['user_encoder']['n_known_ids']:,} users",
    ]
    for name, entry in enc["item_oov_by_split"].items():
        if entry:
            lines.append(
                f"   {name:<5} OOV items {entry['n_oov_items']:>4,}"
                f"  OOV interactions {entry['n_oov_interactions']:>7,} ({entry['oov_share']:.4%})"
            )
    lines.append(rule)
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
