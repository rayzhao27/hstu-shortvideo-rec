"""Smoke-test the KuaiRand -> HSTU path. No official sample data.

    python -m models.smoke
    python -m models.smoke --max-impressions 32 --batch-size 4 --device cpu

Loads Stage 1 ``train_seqs.pkl``, builds one batch, runs Meta's HSTU encoder with
interleaved item/action tokens and relative timestamps, then backward.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.loader import PROCESSED_DIR
from data.protocol import TRAIN_STREAM
from models.dataset import KuaiRandSequenceDataset, collate_kuairand, load_item_vocab
from utils.meta_repo import META_DIR, prepare


def _fail(message: str) -> None:
    print(f"FAIL  {message}")
    raise SystemExit(1)


def _ok(message: str) -> None:
    print(f"ok    {message}")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 smoke test.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-impressions", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-clone", action="store_true")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    info = prepare(clone=not args.no_clone)
    print("official HSTU encoder :", info["hstu_encoder"])
    print("M-FALCON              :", info["m_falcon"])
    print("training objective    :", info["loss"])
    print("official train loop   :", info["train_loop"])
    print("fbgemm                :", info["fbgemm"])
    print("meta dir              :", info["meta_dir"])
    print()

    official_data = list((META_DIR / "tmp").glob("*")) if (META_DIR / "tmp").exists() else []
    if official_data:
        _fail(f"refusing to run: official sample dumps present under {META_DIR / 'tmp'}")
    _ok("no official MovieLens / Amazon dumps on the path")

    seqs = args.processed_dir / "train_seqs.pkl"
    if not seqs.is_file():
        _fail(f"{seqs} missing — run python -m data.preprocess first")

    vocab = load_item_vocab(args.processed_dir)
    dataset = KuaiRandSequenceDataset.from_processed(
        split="train",
        processed_dir=args.processed_dir,
        max_impressions=args.max_impressions,
        stream=TRAIN_STREAM,
    )
    if len(dataset) < args.batch_size:
        _fail(f"only {len(dataset)} train users with a recommended target")
    _ok(f"loaded {len(dataset):,} train users with ≥1 recommended target, vocab={vocab}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_kuairand(rows, args.max_impressions),
    )
    batch = next(iter(loader))
    for key in ("past_ids", "actions", "timestamps", "past_lengths", "supervision"):
        if key not in batch:
            _fail(f"batch missing {key}")

    if int((batch["past_ids"] == 0).sum()) == 0 and int(batch["past_lengths"].min()) < args.max_impressions:
        _fail("expected right-padding with item id 0")
    if not bool(torch.all(batch["past_lengths"] >= 2)):
        _fail("a row has fewer than 2 impressions")

    for i in range(batch["past_ids"].size(0)):
        n = int(batch["past_lengths"][i])
        ts = batch["timestamps"][i, :n]
        if n > 1 and bool((ts[1:] < ts[:-1]).any()):
            _fail(f"row {i}: timestamps decrease inside the valid prefix")
        if bool(batch["supervision"][i, 0] != 0):
            _fail(f"row {i}: position 0 is marked as a target")
        if int(batch["supervision"][i, :n].sum()) == 0:
            _fail(f"row {i}: no recommended target in the window")
        if bool((batch["past_ids"][i, n:] != 0).any()):
            _fail(f"row {i}: padding region is not zeros")
    _ok("batch layout: chronological, right-padded, position 0 unmarked, recommended targets present")

    # Interleaved timestamps must share a time between an item and its action.
    from models.hstu import HSTUOnKuaiRand, interleave_timestamps

    interleaved_ts = interleave_timestamps(batch["timestamps"])
    if interleaved_ts.shape != (args.batch_size, args.max_impressions * 2):
        _fail(f"interleaved timestamps {tuple(interleaved_ts.shape)}")
    if not torch.equal(interleaved_ts[:, 0::2], batch["timestamps"]):
        _fail("item-slot timestamps do not match the impression times")
    if not torch.equal(interleaved_ts[:, 1::2], batch["timestamps"]):
        _fail("action-slot timestamps do not match the impression times")
    _ok("relative-time axis: each impression timestamp is repeated on (item, action)")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARN  --device cuda requested but CUDA is not visible; using cpu")
        device = torch.device("cpu")
    batch = {k: v.to(device) for k, v in batch.items()}

    torch.manual_seed(0)
    model = HSTUOnKuaiRand(
        num_items=vocab,
        max_impressions=args.max_impressions,
        embedding_dim=args.embedding_dim,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.0,
        verbose=False,
    ).to(device)
    model.train()

    encoded = model.encode(batch)
    expect_len = model.token_length + model.max_output_len
    if encoded.shape != (args.batch_size, expect_len, args.embedding_dim):
        _fail(f"encoded shape {tuple(encoded.shape)}, expected {(args.batch_size, expect_len, args.embedding_dim)}")
    if not torch.isfinite(encoded).all():
        _fail("forward produced non-finite hidden states")
    _ok(f"forward  encoded {tuple(encoded.shape)}  device={encoded.device}")

    queries = model.next_item_queries(encoded)
    if queries.shape[1] != args.max_impressions - 1:
        _fail(f"next-item queries cover {queries.shape[1]} steps, expected {args.max_impressions - 1}")
    _ok("next-item queries sit on action tokens (odd positions)")

    loss = model.next_item_loss(batch, encoded)
    if not torch.isfinite(loss):
        _fail(f"loss is {loss}")
    loss_value = float(loss.detach())
    if loss_value <= 0:
        _fail(f"loss {loss_value} should be a positive CE")
    _ok(f"loss     {loss_value:.4f}  (full-softmax next-item CE, recommended targets only)")

    model.zero_grad(set_to_none=True)
    loss.backward()
    checked = 0
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        checked += 1
        if param.grad is None:
            missing.append(name)
            continue
        if not torch.isfinite(param.grad).all():
            _fail(f"non-finite grad on {name}")
    if missing:
        _fail("no grad on: " + ", ".join(missing[:8]))
    item_grad = model.embedding._item_emb.weight.grad
    action_grad = model.hstu._input_features_preproc._action_emb.weight.grad
    if item_grad is None or float(item_grad.norm()) == 0:
        _fail("item embedding received no gradient")
    if action_grad is None or float(action_grad.norm()) == 0:
        _fail("action embedding received no gradient")
    _ok(f"backward {checked} tensors, item-emb grad {float(item_grad.norm()):.4f}, "
        f"action-emb grad {float(action_grad.norm()):.4f}")

    print()
    print("STAGE 2 smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
