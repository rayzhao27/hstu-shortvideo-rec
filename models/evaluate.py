"""Ranking metrics for HSTU on KuaiRand, on both protocol streams.

    python -m models.evaluate --checkpoint runs/base/checkpoints/best.pt --split test

What is measured, precisely:

* At every position the Stage 1 split marks as a target, the model scores the whole
  catalogue and we take the rank of the item the user actually saw next. Recall@K is
  the share of targets ranked in the top K; NDCG@K is ``1/log2(rank+1)`` for those,
  0 otherwise. With exactly one relevant item per target the ideal DCG is 1, so NDCG
  needs no normalisation term.
* **No negative sampling.** The catalogue is ~7.6k items on Pure, so ranking against
  all of it is affordable and removes the sampled-metric bias that makes numbers
  incomparable between papers.
* **Seen items are not filtered out.** 2.5% of KuaiRand impressions are a repeat of a
  video the user already saw, so a repeat is a legitimate next item and masking
  history would make some targets unreachable.
* PAD and UNK are removed from the candidate set: they are not videos.
* Ties are broken pessimistically for the model (an item scoring exactly equal to the
  target does not count against it), which only matters at initialisation.

Two streams, one pass: ranks are computed once for every target and then bucketed by
``is_rand``, so the two stream numbers are guaranteed to come from the same forward
pass and to partition the target set. Per :mod:`data.protocol`, report both and never
average them.

**What the random stream can and cannot measure here.** Stage 1 set up the random
stream (``is_rand=1``) as the unbiased slice, and it is - for predicting a user's
*response* to an item that was shown. It cannot serve as an unbiased quality metric
for *which item comes next*, because on that stream the next item was drawn by the
platform's uniform sampler rather than chosen by a policy or a user. Measured on val:
the random stream's target distribution carries 12.81 bits of entropy against 12.89
for a perfectly uniform draw over the 7,580-item catalogue, and a popularity-ranking
baseline scores Recall@10 = 0.0010 on it, i.e. the 10/7580 = 0.0013 chance rate. No
model can beat chance at guessing a uniform random variable.

So the random-stream retrieval number is reported as a **negative control**, not as a
model score: it is expected to sit at chance, and a value meaningfully above chance
is evidence of leakage rather than of a good model. The headline stays the
``recommended`` stream. ``--baseline`` adds the popularity and chance reference points
next to the model, because on the recommended stream popularity alone already reaches
NDCG@10 = 0.0160, and a model number quoted without that is not interpretable.

Because sequences are left-truncated to a fixed window, some targets of very long
users fall outside it. That fraction is reported as ``coverage`` beside every metric
rather than left implicit - at the default eval window of 256 it is >=99.6%.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loader import PROCESSED_DIR
from data.protocol import EVAL_STREAMS, PRIMARY_EVAL_STREAM
from data.sequences import load_sequences
from models.dataset import KuaiRandSequenceDataset, collate_kuairand, load_item_vocab
from models.hstu import N_RESERVED_ITEM_IDS, HSTUOnKuaiRand

DEFAULT_KS: tuple[int, ...] = (10, 50)
DEFAULT_EVAL_WINDOW = 256
# Rows of (n_targets, vocab) logits held at once. 4096 x 7582 x 4B = 124MB.
SCORE_CHUNK = 4096


class StreamAccumulator:
    """Running Recall@K / NDCG@K / MRR for one stream."""

    def __init__(self, ks: tuple[int, ...]) -> None:
        self.ks = ks
        self.n_targets = 0
        self.hits = dict.fromkeys(ks, 0.0)
        self.ndcg = dict.fromkeys(ks, 0.0)
        self.mrr = 0.0
        self.users: set[int] = set()

    def update(self, ranks: torch.Tensor, users: torch.Tensor) -> None:
        if ranks.numel() == 0:
            return
        ranks = ranks.to(torch.float64)
        self.n_targets += int(ranks.numel())
        self.mrr += float((1.0 / ranks).sum())
        gains = 1.0 / torch.log2(ranks + 1.0)
        for k in self.ks:
            inside = ranks <= k
            self.hits[k] += float(inside.sum())
            self.ndcg[k] += float(gains[inside].sum())
        self.users.update(users.tolist())

    def result(self) -> dict:
        n = max(self.n_targets, 1)
        out: dict = {"n_targets": self.n_targets, "n_users": len(self.users)}
        for k in self.ks:
            out[f"recall@{k}"] = self.hits[k] / n
            out[f"ndcg@{k}"] = self.ndcg[k] / n
        out["mrr"] = self.mrr / n
        return out


@torch.no_grad()
def evaluate(
    model: HSTUOnKuaiRand,
    loader: DataLoader,
    device: torch.device,
    ks: tuple[int, ...] = DEFAULT_KS,
    autocast_dtype: torch.dtype | None = None,
    max_batches: int | None = None,
) -> dict:
    """Rank every marked target against the full catalogue, bucketed by stream."""
    model.eval()
    accs = {name: StreamAccumulator(ks) for name in ("all",) + EVAL_STREAMS}
    n_unrankable = 0

    for n_batch, batch in enumerate(loader):
        if max_batches is not None and n_batch >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        use_amp = autocast_dtype is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            encoded = model.encode(batch)
            queries = model.next_item_queries(encoded)

        # Position i of these shifted views is impression i+1, the item that the
        # query at i has to predict. Same alignment the training loss uses.
        targets = batch["past_ids"][:, 1:]
        marked = batch["supervision"][:, 1:] > 0
        is_rand = batch["is_rand"][:, 1:]
        rows = torch.nonzero(marked, as_tuple=False)
        if rows.numel() == 0:
            continue

        flat_queries = queries.float()[marked]
        flat_targets = targets[marked]
        flat_users = batch["users"][rows[:, 0]]
        flat_is_rand = is_rand[marked]

        # A target below N_RESERVED_ITEM_IDS is PAD or UNK and has no reachable
        # embedding row, so it could never be ranked. k-core filtering makes this
        # empty; counted rather than assumed, and surfaced in the output.
        rankable = flat_targets >= N_RESERVED_ITEM_IDS
        n_unrankable += int((~rankable).sum())
        if not bool(rankable.all()):
            flat_queries = flat_queries[rankable]
            flat_targets = flat_targets[rankable]
            flat_users = flat_users[rankable]
            flat_is_rand = flat_is_rand[rankable]

        ranks = _ranks_against_catalogue(model, flat_queries, flat_targets)
        accs["all"].update(ranks, flat_users)
        for name, flag in (("recommended", 0), ("random", 1)):
            sel = flat_is_rand == flag
            accs[name].update(ranks[sel], flat_users[sel])

    out = {name: acc.result() for name, acc in accs.items()}
    partition = out["recommended"]["n_targets"] + out["random"]["n_targets"]
    if partition != out["all"]["n_targets"]:
        raise AssertionError(
            f"streams do not partition the targets: recommended+random={partition} "
            f"!= all={out['all']['n_targets']}"
        )
    out["n_unrankable_targets"] = n_unrankable
    return out


def _ranks_against_catalogue(
    model: HSTUOnKuaiRand, queries: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """1-based rank of each target item among all real items."""
    table = model.item_table().float()
    ranks = torch.empty(queries.size(0), dtype=torch.int64, device=queries.device)
    for start in range(0, queries.size(0), SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, queries.size(0))
        scores = torch.matmul(queries[start:stop], table.t())
        scores[:, :N_RESERVED_ITEM_IDS] = float("-inf")
        target_score = scores.gather(1, targets[start:stop].unsqueeze(1))
        ranks[start:stop] = 1 + (scores > target_score).sum(dim=1)
    return ranks


def item_popularity(processed_dir: Path) -> np.ndarray:
    """Impression counts per item id over the train split only.

    Train only, for the same reason the item encoder is fit on train only: a
    baseline that peeks at val/test popularity is not a baseline.
    """
    counts = np.zeros(load_item_vocab(processed_dir), dtype=np.int64)
    for record in load_sequences(Path(processed_dir) / "train_seqs.pkl"):
        np.add.at(counts, np.asarray(record["items"], dtype=np.int64), 1)
    counts[:N_RESERVED_ITEM_IDS] = -1
    return counts


def popularity_rank_table(counts: np.ndarray) -> np.ndarray:
    """1-based rank per item id, ties broken the same strict-greater way as the model."""
    order = np.argsort(-counts, kind="stable")
    ranks = np.empty_like(counts)
    ranks[order] = np.arange(1, counts.size + 1)
    return ranks


def evaluate_popularity(
    dataset: KuaiRandSequenceDataset,
    processed_dir: Path,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Rank every target by global train popularity. Same targets, same rank rule."""
    ranks_by_item = popularity_rank_table(item_popularity(processed_dir))
    accs = {name: StreamAccumulator(ks) for name in ("all",) + EVAL_STREAMS}
    for record in dataset.records:
        marked = record["supervision"].astype(bool)
        if not marked.any():
            continue
        items = record["items"][marked]
        is_rand = record["is_rand"][marked]
        keep = items >= N_RESERVED_ITEM_IDS
        items, is_rand = items[keep], is_rand[keep]
        ranks = torch.from_numpy(ranks_by_item[items])
        users = torch.full((items.size,), record["user"], dtype=torch.int64)
        accs["all"].update(ranks, users)
        for name, flag in (("recommended", 0), ("random", 1)):
            sel = torch.from_numpy(is_rand == flag)
            accs[name].update(ranks[sel], users[sel])
    return {name: acc.result() for name, acc in accs.items()}


def chance_metrics(n_real_items: int, ks: tuple[int, ...] = DEFAULT_KS) -> dict:
    """What a uniformly random ranker scores, as the floor every number sits above."""
    ranks = np.arange(1, n_real_items + 1, dtype=np.float64)
    gains = 1.0 / np.log2(ranks + 1.0)
    out: dict = {"n_targets": 0, "n_users": 0}
    for k in ks:
        out[f"recall@{k}"] = k / n_real_items
        out[f"ndcg@{k}"] = float(gains[:k].sum() / n_real_items)
    out["mrr"] = float((1.0 / ranks).mean())
    return out


def build_eval_loader(
    split: str,
    processed_dir: Path,
    max_impressions: int,
    batch_size: int,
    num_workers: int = 0,
    limit_users: int | None = None,
) -> tuple[DataLoader, KuaiRandSequenceDataset]:
    """Loader over ``split`` with every target kept, so both streams are available."""
    dataset = KuaiRandSequenceDataset.from_processed(
        split=split,
        processed_dir=processed_dir,
        max_impressions=max_impressions,
        stream="all",
        require_target=True,
        limit_users=limit_users,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # partial, not a lambda: worker processes have to pickle this.
        collate_fn=partial(collate_kuairand, max_impressions=max_impressions),
    )
    return loader, dataset


def format_metrics(
    metrics: dict,
    ks: tuple[int, ...] = DEFAULT_KS,
    baseline: dict | None = None,
    chance: dict | None = None,
) -> str:
    header = f"{'stream':<24}{'users':>9}{'targets':>11}"
    for k in ks:
        header += f"{'recall@' + str(k):>12}{'ndcg@' + str(k):>11}"
    header += f"{'mrr':>10}"
    lines = [header, "-" * len(header)]

    def row_for(label: str, row: dict, show_counts: bool = True) -> str:
        counts = f"{row['n_users']:>9,}{row['n_targets']:>11,}" if show_counts else " " * 20
        line = f"{label:<24}{counts}"
        for k in ks:
            line += f"{row[f'recall@{k}']:>12.4f}{row[f'ndcg@{k}']:>11.4f}"
        return line + f"{row['mrr']:>10.4f}"

    for name in ("recommended", "random", "all"):
        lines.append(row_for(f"HSTU  {name}", metrics[name]))
        if baseline is not None:
            lines.append(row_for(f"  popularity  {name}", baseline[name], show_counts=False))
    if chance is not None:
        lines.append(row_for("  chance (uniform)", chance, show_counts=False))
    return "\n".join(lines)


def headline(metrics: dict, k: int = 10) -> float:
    """The single number Stage 3 reports, per :mod:`data.protocol`."""
    return float(metrics[PRIMARY_EVAL_STREAM][f"ndcg@{k}"])


def load_checkpoint(path: Path, device: torch.device) -> tuple[HSTUOnKuaiRand, dict]:
    """Rebuild the model from the shape recorded in the checkpoint, then load weights."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = HSTUOnKuaiRand(
        num_items=ckpt["vocab_size"],
        max_impressions=cfg["max_impressions"],
        embedding_dim=cfg["embedding_dim"],
        num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"],
        dropout_rate=cfg["dropout_rate"],
        item_l2_norm=cfg["item_l2_norm"],
        temperature=cfg["temperature"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an HSTU checkpoint on both streams.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=None,
        help="eval window; defaults to the checkpoint's eval_max_impressions",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None, help="write metrics as JSON")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help="skip the popularity and chance reference rows")
    parser.set_defaults(baseline=True)
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    window = args.max_impressions or ckpt["config"].get("eval_max_impressions", DEFAULT_EVAL_WINDOW)

    # The encoder allocates positional/attention buffers for a fixed token axis, so a
    # wider eval window than the trained one needs the buffers rebuilt, not just a
    # different collate width.
    if window != model.max_impressions:
        raise SystemExit(
            f"checkpoint was built for max_impressions={model.max_impressions} but "
            f"--max-impressions={window} was requested. Re-run training with "
            f"--eval-max-impressions {window} (train.py sizes the encoder to the "
            f"larger of the two windows)."
        )

    loader, dataset = build_eval_loader(
        split=args.split,
        processed_dir=args.processed_dir,
        max_impressions=window,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"{args.split}: {len(dataset):,} users, eval window {window} impressions")

    ks = tuple(args.ks)
    metrics = evaluate(model, loader, device, ks=ks)
    metrics["split"] = args.split
    metrics["eval_max_impressions"] = window
    metrics["coverage"] = dataset.coverage()
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["step"] = ckpt.get("global_step")

    baseline = chance = None
    if args.baseline:
        baseline = evaluate_popularity(dataset, args.processed_dir, ks)
        chance = chance_metrics(ckpt["vocab_size"] - N_RESERVED_ITEM_IDS, ks)
        metrics["popularity_baseline"] = baseline
        metrics["chance"] = chance

    print()
    print(format_metrics(metrics, ks, baseline, chance))
    if 10 in ks:
        print(f"\nheadline: {PRIMARY_EVAL_STREAM} ndcg@10 = {headline(metrics):.4f}")
    print()
    for stream in EVAL_STREAMS:
        cov = metrics["coverage"][stream]
        print(f"coverage {stream:<12} {cov['targets_in_window']:>9,} / "
              f"{cov['targets_in_split']:>9,} targets in window ({cov['fraction']:.2%})")
    if metrics["n_unrankable_targets"]:
        print(f"WARNING: {metrics['n_unrankable_targets']:,} targets were PAD/UNK and skipped")
    print("\nreminder: the random stream is a negative control, not a model score - its "
          "targets\nwere drawn by a uniform sampler, so chance is the expected value "
          "there (see module docstring).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
