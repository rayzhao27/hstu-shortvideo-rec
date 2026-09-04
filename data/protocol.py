"""The training and evaluation protocol, in one place.

    python -m data.protocol      # print the target counts every model will be scored on

HSTU and the BERT4Rec / DIN baselines must be trained and scored on identical targets,
otherwise the comparison measures the protocol as much as the model. Rather than
trusting three training scripts to agree, they all call :func:`target_mask` here.

The agreed protocol:

* **Training** scores the ``recommended`` stream only (``is_rand=0``), so the training
  objective matches the policy that serves production traffic.
* **Evaluation** reports two numbers. The headline metric is the ``recommended``
  stream, which is what an online A/B test would move. The second is the ``random``
  stream (``is_rand=1``), an unbiased sample of user response, which says how much of
  the headline number the model inherited from the logging policy rather than learned.

Both streams come out of one preprocessing run. ``data/preprocess.py`` keeps
``--target-policy all`` so every eval-window impression stays in the file, and the
stream is selected here by masking. This is exactly equivalent to baking the policy in
at preprocessing time - :func:`data.sequences.build_split_sequences` masks positions
instead of dropping users specifically to preserve that equivalence - and it removes
any chance of two models being compared on artifacts from two different runs.

Why the two numbers are not comparable to each other: the random log only covers
2022-04-22 onwards and the standard log thins out over the same window, so the test
window is 88% random exposure. The two streams therefore have very different sample
sizes, and the random stream draws from a much wider slice of the catalogue than the
production policy would ever show. Report both, explain the gap, never average them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.loader import PROCESSED_DIR
from data.sequences import load_sequences
from data.splitting import SPLIT_NAMES

TRAIN_STREAM = "recommended"
PRIMARY_EVAL_STREAM = "recommended"
EVAL_STREAMS: tuple[str, ...] = ("recommended", "random")

# None means "do not filter by exposure".
STREAM_IS_RAND: dict[str, int | None] = {"all": None, "recommended": 0, "random": 1}


def target_mask(record: dict, stream: str = "all") -> np.ndarray:
    """Positions of ``record`` that ``stream`` scores.

    Never widens the split's own target mask, so this cannot resurrect an interaction
    the split was not supposed to see.
    """
    if stream not in STREAM_IS_RAND:
        raise ValueError(f"unknown stream {stream!r}, expected one of {sorted(STREAM_IS_RAND)}")
    flag = STREAM_IS_RAND[stream]
    if flag is None:
        return record["is_target"]
    return record["is_target"] & (record["is_rand"] == flag)


def stream_summary(records: list[dict], stream: str) -> dict:
    """How many users and targets a stream contributes, i.e. its sample size."""
    masks = [target_mask(record, stream) for record in records]
    per_user = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    scored = per_user > 0
    return {
        "stream": stream,
        "n_users": int(scored.sum()),
        "n_targets": int(per_user.sum()),
        "targets_per_scored_user": float(per_user[scored].mean()) if scored.any() else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the evaluation protocol.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args(argv)

    stats_path = args.processed_dir / "preprocess_stats.json"
    if stats_path.exists():
        config = json.loads(stats_path.read_text())["config"]
        if config.get("target_policy") != "all":
            print(f"WARNING: these files were built with --target-policy "
                  f"{config['target_policy']!r}, so one of the two streams is missing. "
                  f"Rebuild with the default 'all'.\n")

    print(f"train on : {TRAIN_STREAM}")
    print(f"report   : {', '.join(EVAL_STREAMS)}  (headline = {PRIMARY_EVAL_STREAM})\n")
    print(f"{'split':<7}{'stream':<14}{'users':>9}{'targets':>12}{'targets/user':>15}")
    print("-" * 57)

    for split in SPLIT_NAMES:
        records = load_sequences(args.processed_dir / f"{split}_seqs.pkl")
        for stream in ("all",) + EVAL_STREAMS:
            summary = stream_summary(records, stream)
            print(f"{split:<7}{stream:<14}{summary['n_users']:>9,}"
                  f"{summary['n_targets']:>12,}{summary['targets_per_scored_user']:>15.1f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
