"""Stage 0: acquire KuaiRand, validate the schema, print the dataset profile.

Run from the repo root:

    python -m data.explore                       # standard + random logs
    python -m data.explore --splits standard     # only the standard log
    python -m data.explore --no-plots --no-cache

Outputs:
    datasets/processed/stats.json          the profile, machine readable
    datasets/processed/log_*.parquet       cached logs (CSV parsing takes ~10x longer)
    pictures/stage0_overview_*.png     overview figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from data import loader, stats
from data.download import DATASETS, RAW_DIR, ensure_dataset
from data.loader import PROCESSED_DIR

logger = logging.getLogger("data.explore")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 dataset profiling.")
    parser.add_argument("--dataset", default="KuaiRand-Pure", choices=sorted(DATASETS))
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--pictures-dir", type=Path, default=Path("pictures"))
    parser.add_argument("--splits", nargs="+", default=["standard", "random"],
                        choices=list(loader.SPLITS))
    parser.add_argument("--sample-rows", type=int, default=30,
                        help="rows of the sampled user timeline to print")
    parser.add_argument("--no-download", dest="download", action="store_false",
                        help="fail instead of downloading if the dataset is missing")
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--no-cache", dest="cache", action="store_false",
                        help="skip writing the parquet cache")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Logs go to stdout so they interleave with the printed report in order.
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    data_dir = ensure_dataset(DATASETS[args.dataset], args.raw_dir, download=args.download)

    stats_by_split = {}
    for split in args.splits:
        logger.info("Loading %s log", split)
        log = loader.load_log(data_dir, split)

        print(f"\n--- {split}: schema ---")
        print(
            pd.DataFrame(
                {
                    "dtype": log.dtypes.astype(str),
                    "non_null": log.notna().sum(),
                    "n_unique": log.nunique(),
                    "min": [log[c].min() for c in log.columns],
                    "max": [log[c].max() for c in log.columns],
                }
            ).to_string()
        )

        split_stats = stats.compute_stats(log, split)
        stats_by_split[split] = split_stats
        print()
        print(stats.format_report(split_stats))

        if split == "standard" and args.sample_rows:
            sample = stats.sample_user_timeline(log, n_rows=args.sample_rows)
            print(
                f"\n--- sample user {sample.user_id}: {sample.n_impressions} impressions in "
                f"{sample.n_sessions} sessions (>30min gap starts a session) ---"
            )
            print(sample.timeline.to_string())

        if args.cache:
            loader.write_parquet(log, args.processed_dir / f"log_{split}.parquet")

        if args.plots:
            from visualization.save_plots import save_log_overview

            save_log_overview(log, args.pictures_dir, split)

        del log

    for name in ("user_features", "video_features_basic"):
        try:
            features = loader.load_features(data_dir, name)
        except FileNotFoundError:
            logger.warning("%s not found in %s, skipping", name, data_dir)
            continue
        if args.cache:
            loader.write_parquet(features, args.processed_dir / f"{name}.parquet")

    stats_path = stats.write_stats(stats_by_split, args.processed_dir / "stats.json")
    logger.info("Stats written to %s", stats_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
