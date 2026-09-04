"""Independent audit of the Stage 1 outputs.

    python -m data.verify

Everything here re-derives its conclusions from the written files rather than
trusting the pipeline that produced them, so a refactor that quietly breaks an
invariant fails loudly here. Exits non-zero if any check fails.

Checks:
  1. record integrity   - array lengths agree, timestamps non-decreasing, the target
                          mask is consistent with n_targets, every target has history
  2. vocabulary         - item/user indices are inside the encoder range and PAD/UNK
                          never appear as a real interaction
  3. prefix containment - a user's train sequence is a byte-exact prefix of their val
                          sequence, and val of test; if it is not, some interaction
                          was reordered or dropped between splits
  4. temporal windows   - for the temporal strategy, every train interaction predates
                          the val boundary and every test target follows the test
                          boundary
  5. flat table         - the sequence files and interactions.parquet agree on totals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data.actions import ACTION_IDS
from data.encoders import N_RESERVED, IdEncoder
from data.loader import PROCESSED_DIR
from data.schema import DATASET_TZ
from data.sequences import load_sequences
from data.splitting import SPLIT_NAMES


class Auditor:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.checks += 1
        if ok:
            print(f"  PASS  {label}")
        else:
            print(f"  FAIL  {label}{': ' + detail if detail else ''}")
            self.failures.append(label)
        return ok


def _ms_to_local(value: int) -> pd.Timestamp:
    return pd.Timestamp(int(value), unit="ms", tz="UTC").tz_convert(DATASET_TZ)


def audit_records(audit: Auditor, split: str, records: list[dict]) -> None:
    bad_length, bad_order, bad_mask, no_history = 0, 0, 0, 0

    for record in records:
        n = len(record["items"])
        if not all(
            len(record[key]) == n
            for key in ("actions", "timestamps", "is_rand", "is_target")
        ):
            bad_length += 1
            continue
        if np.any(np.diff(record["timestamps"]) < 0):
            bad_order += 1
        if int(record["is_target"].sum()) != record["n_targets"] or record["n_targets"] == 0:
            bad_mask += 1
        if not record["is_target"].any() or int(np.argmax(record["is_target"])) < 1:
            no_history += 1

    audit.check(bad_length == 0, f"{split}: array lengths agree", f"{bad_length} records")
    audit.check(bad_order == 0, f"{split}: timestamps non-decreasing", f"{bad_order} records")
    audit.check(bad_mask == 0, f"{split}: target mask matches n_targets", f"{bad_mask} records")
    audit.check(no_history == 0, f"{split}: every target has history", f"{no_history} records")


def audit_vocabulary(
    audit: Auditor, split: str, records: list[dict], item_encoder: IdEncoder, user_encoder: IdEncoder
) -> None:
    max_item = max(int(r["items"].max()) for r in records)
    min_item = min(int(r["items"].min()) for r in records)
    max_user = max(r["user"] for r in records)
    min_user = min(r["user"] for r in records)
    bad_actions = sum(
        int((~np.isin(r["actions"], ACTION_IDS)).sum()) for r in records
    )

    audit.check(
        N_RESERVED <= min_item and max_item < item_encoder.vocab_size,
        f"{split}: item indices inside vocabulary",
        f"range [{min_item}, {max_item}], vocab {item_encoder.vocab_size}",
    )
    audit.check(
        N_RESERVED <= min_user and max_user < user_encoder.vocab_size,
        f"{split}: user indices inside vocabulary",
        f"range [{min_user}, {max_user}], vocab {user_encoder.vocab_size}",
    )
    audit.check(bad_actions == 0, f"{split}: action ids are valid", f"{bad_actions} positions")


def audit_protocol(audit: Auditor, split: str, records: list[dict]) -> None:
    """The two evaluation streams must partition the split's targets exactly.

    If they did not, restricting the stream at read time would stop being equivalent to
    restricting it during preprocessing, and the headline and unbiased metrics would no
    longer be computed over one consistent artifact.
    """
    from data.protocol import EVAL_STREAMS, target_mask

    overlap, missing, scored_head = 0, 0, 0
    for record in records:
        masks = [target_mask(record, stream) for stream in EVAL_STREAMS]
        union = np.logical_or.reduce(masks)
        intersection = np.logical_and.reduce(masks)
        overlap += int(intersection.sum())
        missing += int((record["is_target"] & ~union).sum())
        scored_head += int(record["is_target"][0])

    audit.check(overlap == 0, f"{split}: the two streams do not overlap", f"{overlap} positions")
    audit.check(
        missing == 0,
        f"{split}: the two streams cover every target",
        f"{missing} positions in neither",
    )
    audit.check(
        scored_head == 0,
        f"{split}: position 0 is never a target",
        f"{scored_head} records",
    )


def audit_prefix_containment(audit: Auditor, by_split: dict) -> None:
    """A user's history in an earlier split must be an exact prefix of the later one.

    This is the check that would catch a reordering bug: if any interaction moved,
    was duplicated, or leaked backwards between splits, the prefixes stop matching.
    """
    indexed = {
        split: {r["user"]: r for r in records} for split, records in by_split.items()
    }

    for earlier, later in (("train", "val"), ("val", "test")):
        mismatches, compared = 0, 0
        for user, early_record in indexed.get(earlier, {}).items():
            late_record = indexed.get(later, {}).get(user)
            if late_record is None:
                continue
            compared += 1
            n = len(early_record["items"])
            if n > len(late_record["items"]):
                mismatches += 1
                continue
            same = np.array_equal(early_record["items"], late_record["items"][:n]) and np.array_equal(
                early_record["timestamps"], late_record["timestamps"][:n]
            )
            if not same:
                mismatches += 1
        audit.check(
            mismatches == 0,
            f"{earlier} sequence is a prefix of {later} ({compared:,} users compared)",
            f"{mismatches} mismatches",
        )


def audit_temporal_windows(audit: Auditor, by_split: dict, stats: dict) -> None:
    split_stats = stats["split"]
    if split_stats["strategy"] != "temporal":
        print("  SKIP  temporal window bounds (strategy is "
              f"{split_stats['strategy']}, which overlaps in time by design)")
        return

    boundaries = split_stats["boundaries"]
    val_start = pd.Timestamp(boundaries["val_start"]).value // 1_000_000
    test_start = pd.Timestamp(boundaries["test_start"]).value // 1_000_000

    train_max = max(int(r["timestamps"].max()) for r in by_split["train"])
    audit.check(
        train_max < val_start,
        "train holds nothing at or after the val boundary",
        f"last train interaction {_ms_to_local(train_max)}",
    )

    val_targets = np.concatenate([r["timestamps"][r["is_target"]] for r in by_split["val"]])
    audit.check(
        bool((val_targets >= val_start).all() and (val_targets < test_start).all()),
        "every val target falls inside the val window",
    )

    test_targets = np.concatenate([r["timestamps"][r["is_target"]] for r in by_split["test"]])
    audit.check(
        bool((test_targets >= test_start).all()),
        "every test target falls inside the test window",
    )

    val_history_max = max(int(r["timestamps"].max()) for r in by_split["val"])
    audit.check(
        val_history_max < test_start,
        "val holds nothing at or after the test boundary",
        f"last val interaction {_ms_to_local(val_history_max)}",
    )


def audit_against_flat_table(audit: Auditor, by_split: dict, processed_dir: Path) -> None:
    """The test file sees every split, so it must reproduce the flat table exactly.

    Only users dropped for having no test target may be missing; for everyone else the
    item order and the timestamps have to match row for row.
    """
    path = processed_dir / "interactions.parquet"
    if not path.exists():
        print(f"  SKIP  cross-check against {path} (not found)")
        return

    flat = pd.read_parquet(path, columns=["user_idx", "item_idx", "time_ms"])
    test_records = {r["user"]: r for r in by_split["test"]}

    grouped = flat.groupby("user_idx", sort=False)
    length_mismatch, content_mismatch, compared = 0, 0, 0

    for user, group in grouped:
        record = test_records.get(int(user))
        if record is None:
            continue
        compared += 1
        if len(group) != len(record["items"]):
            length_mismatch += 1
            continue
        if not (
            np.array_equal(group["item_idx"].to_numpy(), record["items"])
            and np.array_equal(group["time_ms"].to_numpy(), record["timestamps"])
        ):
            content_mismatch += 1

    audit.check(
        compared == len(test_records),
        "every test user appears in the flat table",
        f"{compared:,} of {len(test_records):,}",
    )
    audit.check(
        length_mismatch == 0,
        f"test sequence lengths match the flat table ({compared:,} users)",
        f"{length_mismatch} users",
    )
    audit.check(
        content_mismatch == 0,
        "test items and timestamps match the flat table row for row",
        f"{content_mismatch} users",
    )


def audit_round_trip(audit: Auditor, by_split: dict, processed_dir: Path, n_users: int) -> None:
    """Decode remapped ids back to raw ids and compare against the original CSVs.

    The other checks all read artifacts the pipeline wrote. This one goes back to the
    source, so it is the only check that can catch a mapping applied in the wrong
    direction or a sort that silently reordered a user's history.
    """
    from data.download import DATASETS, ensure_dataset
    from data.loader import load_log
    from data.preprocess import DEDUPE_KEYS, SORT_KEYS

    item_encoder = IdEncoder.load(processed_dir / "item_encoder.pkl")
    user_encoder = IdEncoder.load(processed_dir / "user_encoder.pkl")
    stats = json.loads((processed_dir / "preprocess_stats.json").read_text())

    data_dir = ensure_dataset(DATASETS[stats["config"]["dataset"]])
    logs = [load_log(data_dir, "standard", with_timestamp=False, sort=False)]
    if stats["config"]["include_random"]:
        logs.append(load_log(data_dir, "random", with_timestamp=False, sort=False))
    columns = ["user_id", "video_id", "time_ms", "is_rand", "play_time_ms"]
    raw = pd.concat([log[columns] for log in logs], ignore_index=True)

    # Replay the same cleaning the pipeline applies before any remapping happens.
    raw = raw.sort_values("play_time_ms", ascending=False, kind="mergesort")
    raw = raw.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    raw = raw.sort_values(SORT_KEYS, kind="mergesort")

    rng = np.random.default_rng(0)
    records = by_split["test"]
    sample = [records[i] for i in rng.choice(len(records), size=min(n_users, len(records)),
                                             replace=False)]
    by_user = {int(u): g for u, g in raw.groupby("user_id", sort=False)}

    mismatches = 0
    for record in sample:
        raw_user = int(user_encoder.idx_to_raw[record["user"]])
        raw_rows = by_user.get(raw_user)
        decoded_items = item_encoder.idx_to_raw[record["items"]]

        # Filtering removes interactions, so the decoded sequence is a subsequence of
        # the raw one; equality of the kept rows is what matters.
        raw_items = raw_rows["video_id"].to_numpy()
        raw_times = raw_rows["time_ms"].to_numpy()
        keep = np.isin(raw_items, decoded_items)
        if not (
            np.array_equal(raw_items[keep], decoded_items)
            and np.array_equal(raw_times[keep], record["timestamps"])
        ):
            mismatches += 1

    audit.check(
        mismatches == 0,
        f"decoded sequences match the raw CSVs ({len(sample)} users sampled)",
        f"{mismatches} mismatches",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit the Stage 1 outputs.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--deep", action="store_true",
                        help="also reload the raw CSVs and round-trip a sample of users")
    parser.add_argument("--deep-users", type=int, default=200)
    args = parser.parse_args(argv)

    stats_path = args.processed_dir / "preprocess_stats.json"
    if not stats_path.exists():
        print(f"{stats_path} not found; run python -m data.preprocess first", file=sys.stderr)
        return 1

    stats = json.loads(stats_path.read_text())
    item_encoder = IdEncoder.load(args.processed_dir / "item_encoder.pkl")
    user_encoder = IdEncoder.load(args.processed_dir / "user_encoder.pkl")
    by_split = {
        split: load_sequences(args.processed_dir / f"{split}_seqs.pkl") for split in SPLIT_NAMES
    }

    audit = Auditor()
    print(f"auditing {args.processed_dir} "
          f"(strategy={stats['split']['strategy']}, "
          f"target policy={stats.get('target_policy', {}).get('policy', 'all')})\n")

    print("record integrity")
    for split, records in by_split.items():
        audit_records(audit, split, records)

    print("\nvocabulary")
    for split, records in by_split.items():
        audit_vocabulary(audit, split, records, item_encoder, user_encoder)

    print("\nevaluation protocol")
    for split, records in by_split.items():
        audit_protocol(audit, split, records)

    print("\nprefix containment")
    audit_prefix_containment(audit, by_split)

    print("\ntemporal windows")
    audit_temporal_windows(audit, by_split, stats)

    print("\nflat table")
    audit_against_flat_table(audit, by_split, args.processed_dir)

    print("\nround trip to the raw CSVs")
    if args.deep:
        audit_round_trip(audit, by_split, args.processed_dir, args.deep_users)
    else:
        print("  SKIP  pass --deep to reload the raw logs and check a sample of users")

    print(f"\n{audit.checks - len(audit.failures)}/{audit.checks} checks passed")
    if audit.failures:
        print("failed: " + ", ".join(audit.failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
