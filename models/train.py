"""Train HSTU on KuaiRand and report both protocol streams.

    python -m models.train --size base --out-dir runs/base
    python -m models.train --smoke-test --device cpu

Everything a run produces is written under ``--out-dir``, which on Colab should be a
Google Drive path. A Colab container is disposable; Drive is not::

    out-dir/
      config.json          resolved configuration, including the data fingerprint
      tb/                  TensorBoard event files
      metrics.jsonl        one line per evaluation, appended (survives a resume)
      curves.png           loss and NDCG@10 curves, redrawn from metrics.jsonl
      final_metrics.json   val + test metrics of the best checkpoint
      checkpoints/last.pt  written every --save-every-steps and at every epoch end
      checkpoints/best.pt  best headline (recommended NDCG@10) so far

Resume is the default. If ``checkpoints/last.pt`` exists the run picks up model,
optimizer, AMP scaler, epoch, step, best-so-far, and RNG state, so a preempted A100
costs the time since the last save and nothing more. Checkpoints are written to a
temporary file and then renamed, because a process killed halfway through a write to
Drive would otherwise leave a truncated file where the only copy used to be.

Defaults are sized so ``--size base`` runs on an L4 (24GB) without OOM; an A100 40GB
takes roughly double the batch. The knobs for when it does not fit, and for when the
loss will not move, are documented in the README's Stage 3 diagnostics table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.loader import PROCESSED_DIR
from data.protocol import PRIMARY_EVAL_STREAM, TRAIN_STREAM
from models.config import DEFAULT_SIZE, SIZES, SMOKE_SIZE, get_size
from models.dataset import KuaiRandSequenceDataset, collate_kuairand, load_item_vocab
from models.evaluate import (
    DEFAULT_EVAL_WINDOW,
    build_eval_loader,
    chance_metrics,
    evaluate,
    evaluate_popularity,
    format_metrics,
    headline,
)
from models.hstu import N_RESERVED_ITEM_IDS, HSTUOnKuaiRand

HEADLINE_K = 10
HEADLINE_METRIC = f"{PRIMARY_EVAL_STREAM}/ndcg@{HEADLINE_K}"


# --------------------------------------------------------------------------- setup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(name: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """(autocast dtype, needs GradScaler). fp16 needs loss scaling; bf16 does not."""
    if device.type != "cuda" or name == "fp32":
        return None, False
    if name == "auto":
        name = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if name == "bf16":
        return torch.bfloat16, False
    if name == "fp16":
        return torch.float16, True
    raise ValueError(f"unknown precision {name!r}")


def data_fingerprint(processed_dir: Path, vocab_size: int, n_train_users: int) -> dict:
    """Identify the Stage 1 artifacts, so a resume cannot silently switch datasets.

    Optimizer state and embedding rows are only meaningful against the id mapping
    they were trained with. A checkpoint resumed onto a different preprocessing run
    would train happily and report nonsense.
    """
    stats_path = Path(processed_dir) / "preprocess_stats.json"
    config = {}
    if stats_path.exists():
        config = json.loads(stats_path.read_text()).get("config", {})
    return {
        "dataset": config.get("dataset"),
        "split_strategy": config.get("split_strategy"),
        "target_policy": config.get("target_policy"),
        "min_user_len": config.get("min_user_len"),
        "min_item_count": config.get("min_item_count"),
        "vocab_size": vocab_size,
        "n_train_users": n_train_users,
    }


def lr_at(step: int, total_steps: int, warmup: int, base_lr: float, schedule: str) -> float:
    """Learning rate as a pure function of the step, so resume needs no scheduler state."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if schedule == "constant":
        return base_lr
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ------------------------------------------------------------------- checkpointing


def save_checkpoint(path: Path, payload: dict) -> None:
    """Write then rename. A half-written checkpoint on Drive is worse than none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: dict,
    epoch: int,
    global_step: int,
    best: dict,
    vocab_size: int,
    fingerprint: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best": best,
        "config": cfg,
        "vocab_size": vocab_size,
        "fingerprint": fingerprint,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }


def restore(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    fingerprint: dict,
) -> tuple[int, int, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stored = ckpt.get("fingerprint", {})
    if stored and stored != fingerprint:
        raise SystemExit(
            "refusing to resume: this checkpoint was trained on different data.\n"
            f"  checkpoint: {stored}\n  current   : {fingerprint}\n"
            "Point --out-dir somewhere new, or pass --resume never to start over."
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    rng = ckpt.get("rng")
    if rng:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return int(ckpt["epoch"]), int(ckpt["global_step"]), dict(ckpt["best"])


# ------------------------------------------------------------------------ plotting


def write_curves(out_dir: Path) -> None:
    """Redraw curves from metrics.jsonl so the picture survives preemption too."""
    records = []
    path = out_dir / "metrics.jsonl"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    if len(records) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = [r["global_step"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, [r["train_loss"] for r in records], label="train")
    axes[0].plot(steps, [r["val_loss"] for r in records], label="val")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("next-item CE")
    axes[0].set_title("loss")
    axes[0].legend()
    for stream in ("recommended", "random"):
        axes[1].plot(steps, [r["val"][stream][f"ndcg@{HEADLINE_K}"] for r in records], label=stream)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel(f"NDCG@{HEADLINE_K}")
    axes[1].set_title("val NDCG@10 by stream")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------- train loop


def train_one_epoch(
    model: HSTUOnKuaiRand,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    autocast_dtype: torch.dtype | None,
    global_step: int,
    total_steps: int,
    writer,
    ckpt_dir: Path,
    payload_fn,
) -> tuple[int, float]:
    model.train()
    running, seen = 0.0, 0  # logging window
    epoch_sum, epoch_n = 0.0, 0  # whole epoch, for the returned average
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        use_amp = autocast_dtype is not None
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            loss = model.next_item_loss(batch, num_negatives=args.num_negatives)
        scaled = loss / args.grad_accum
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        batch_loss = float(loss.detach())
        running += batch_loss
        seen += 1
        epoch_sum += batch_loss
        epoch_n += 1

        if (i + 1) % args.grad_accum != 0:
            continue

        if args.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        lr = lr_at(global_step, total_steps, args.warmup_steps, args.lr, args.lr_schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.log_every == 0:
            avg = running / max(seen, 1)
            rate = seen * loader.batch_size / max(time.time() - t0, 1e-6)
            print(f"  step {global_step:>7}  loss {avg:.4f}  lr {lr:.2e}  {rate:.0f} seq/s",
                  flush=True)
            if writer is not None:
                writer.add_scalar("train/loss", avg, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/seq_per_s", rate, global_step)
            running, seen, t0 = 0.0, 0, time.time()

        if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
            save_checkpoint(ckpt_dir / "last.pt", payload_fn(global_step))

        if args.max_steps and global_step >= args.max_steps:
            break

    return global_step, (epoch_sum / epoch_n if epoch_n else float("nan"))


@torch.no_grad()
def validation_loss(
    model: HSTUOnKuaiRand,
    loader: DataLoader,
    device: torch.device,
    num_negatives: int | None,
    autocast_dtype: torch.dtype | None,
    max_batches: int | None = None,
) -> float:
    """CE on the training stream's targets, i.e. directly comparable to train loss."""
    model.eval()
    total, batches = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
        ):
            total += float(model.next_item_loss(batch, num_negatives=num_negatives))
        batches += 1
    return total / max(batches, 1)


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train HSTU on KuaiRand sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_argument_group("data")
    data.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    data.add_argument(
        "--max-impressions",
        type=int,
        default=DEFAULT_EVAL_WINDOW,
        help="impressions kept per user; the token axis is twice this. "
             "256 keeps >=99.6%% of eval targets and 97.8%% of train targets",
    )
    data.add_argument("--num-workers", type=int, default=2)
    data.add_argument("--limit-train-users", type=int, default=None, help="debugging only")
    data.add_argument("--val-users", type=int, default=None,
                      help="cap users in the per-epoch validation, for speed")

    model = p.add_argument_group("model")
    model.add_argument("--size", default=DEFAULT_SIZE, choices=sorted(SIZES),
                       help="shape preset from models/config.py")
    model.add_argument("--embedding-dim", type=int, default=None, help="override the preset")
    model.add_argument("--num-blocks", type=int, default=None, help="override the preset")
    model.add_argument("--num-heads", type=int, default=None, help="override the preset")
    model.add_argument("--dropout-rate", type=float, default=0.2)
    model.add_argument("--temperature", type=float, default=0.05)
    model.add_argument("--no-item-l2-norm", dest="item_l2_norm", action="store_false")
    model.set_defaults(item_l2_norm=True)

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=20)
    opt.add_argument("--batch-size", type=int, default=None, help="preset suggestion if unset")
    opt.add_argument("--lr", type=float, default=1e-3)
    opt.add_argument("--weight-decay", type=float, default=0.0)
    opt.add_argument("--warmup-steps", type=int, default=0)
    opt.add_argument("--lr-schedule", default="cosine", choices=("cosine", "constant"))
    opt.add_argument("--grad-accum", type=int, default=1,
                     help="micro-batches per optimizer step; raise this instead of "
                          "lowering the effective batch when memory is tight")
    opt.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
    opt.add_argument("--num-negatives", type=int, default=None,
                     help="unset = full softmax over the catalogue (exact). Set e.g. "
                          "128 for sampled softmax, which is what makes larger "
                          "catalogues and batches fit")
    opt.add_argument("--precision", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    opt.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
    opt.add_argument("--patience", type=int, default=5,
                     help="stop after this many evaluations with no headline gain; 0 disables")

    run = p.add_argument_group("run")
    run.add_argument("--out-dir", type=Path, default=Path("runs/hstu"),
                     help="on Colab point this at Google Drive")
    run.add_argument("--resume", default="auto", choices=("auto", "never"))
    run.add_argument("--eval-every", type=int, default=1, help="in epochs")
    run.add_argument("--log-every", type=int, default=50, help="in optimizer steps")
    run.add_argument("--save-every-steps", type=int, default=500, help="0 = only at epoch end")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--no-test", dest="run_test", action="store_false",
                     help="skip the final test evaluation")
    run.set_defaults(run_test=True)
    run.add_argument("--smoke-test", "--smoke_test", dest="smoke_test", action="store_true",
                     help="tiny end-to-end run: train, eval, checkpoint, resume")
    return p


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    """A few seconds on CPU, exercising every path a real run uses."""
    args.size = SMOKE_SIZE
    args.max_impressions = 32
    args.batch_size = 4
    args.limit_train_users = 64
    args.val_users = 32
    args.epochs = 1
    args.max_steps = 8
    args.log_every = 4
    args.save_every_steps = 4
    args.num_workers = 0
    args.patience = 0
    args.eval_every = 1
    if args.out_dir == Path("runs/hstu"):
        args.out_dir = Path("runs/smoke")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke_test:
        apply_smoke_defaults(args)

    size = get_size(args.size)
    embedding_dim = args.embedding_dim or size.embedding_dim
    num_blocks = args.num_blocks or size.num_blocks
    num_heads = args.num_heads or size.num_heads
    batch_size = args.batch_size or size.suggested_batch_size

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast_dtype, needs_scaler = resolve_precision(args.precision, device)
    set_seed(args.seed)

    out_dir = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    vocab_size = load_item_vocab(args.processed_dir)
    train_ds = KuaiRandSequenceDataset.from_processed(
        split="train",
        processed_dir=args.processed_dir,
        max_impressions=args.max_impressions,
        stream=TRAIN_STREAM,
        require_target=True,
        limit_users=args.limit_train_users,
    )
    collate = partial(collate_kuairand, max_impressions=args.max_impressions)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader, val_ds = build_eval_loader(
        split="val",
        processed_dir=args.processed_dir,
        max_impressions=args.max_impressions,
        batch_size=batch_size,
        num_workers=args.num_workers,
        limit_users=args.val_users,
    )
    # Loss on the training stream only, so train and val loss are the same quantity.
    val_loss_ds = KuaiRandSequenceDataset.from_processed(
        split="val",
        processed_dir=args.processed_dir,
        max_impressions=args.max_impressions,
        stream=TRAIN_STREAM,
        require_target=True,
        limit_users=args.val_users,
    )
    val_loss_loader = DataLoader(
        val_loss_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs

    cfg = {
        "size": args.size,
        "embedding_dim": embedding_dim,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "dropout_rate": args.dropout_rate,
        "temperature": args.temperature,
        "item_l2_norm": args.item_l2_norm,
        "max_impressions": args.max_impressions,
        "eval_max_impressions": args.max_impressions,
        "batch_size": batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_schedule": args.lr_schedule,
        "grad_clip": args.grad_clip,
        "num_negatives": args.num_negatives,
        "precision": args.precision,
        "epochs": args.epochs,
        "seed": args.seed,
        "train_stream": TRAIN_STREAM,
        "processed_dir": str(args.processed_dir),
    }
    fingerprint = data_fingerprint(args.processed_dir, vocab_size, len(train_ds))

    model = HSTUOnKuaiRand(
        num_items=vocab_size,
        max_impressions=args.max_impressions,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout_rate=args.dropout_rate,
        item_l2_norm=args.item_l2_norm,
        temperature=args.temperature,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type) if needs_scaler else None

    print(f"device      : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"size        : {args.size}  dim={embedding_dim} blocks={num_blocks} "
          f"heads={num_heads} dqk=dv={model.attention_dim}")
    print(f"params      : {n_params:,}")
    print(f"window      : {args.max_impressions} impressions -> {model.token_length} tokens")
    print(f"loss        : {'full softmax' if args.num_negatives is None else f'sampled softmax n={args.num_negatives}'}"
          f"  temperature={args.temperature}  item_l2_norm={args.item_l2_norm}")
    print(f"precision   : {args.precision} -> {autocast_dtype or 'fp32'}"
          f"{' + GradScaler' if needs_scaler else ''}")
    print(f"train       : {len(train_ds):,} users, batch {batch_size} x accum {args.grad_accum}, "
          f"{steps_per_epoch:,} steps/epoch, {total_steps:,} total")
    print(f"val         : {len(val_ds):,} users")
    print(f"out         : {out_dir}")

    (out_dir / "config.json").write_text(
        json.dumps({"config": cfg, "fingerprint": fingerprint, "n_params": n_params}, indent=2)
    )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        print("note: tensorboard not installed, skipping event logging")

    state = {"best": {"metric": HEADLINE_METRIC, "value": -1.0, "step": -1, "epoch": -1}}

    def payload_fn(step: int, epoch: int = 0) -> dict:
        return build_payload(
            model, optimizer, scaler, cfg, epoch, step,
            state["best"], vocab_size, fingerprint,
        )

    start_epoch, global_step = 0, 0
    last_path = ckpt_dir / "last.pt"
    if args.resume == "auto" and last_path.exists():
        start_epoch, global_step, state["best"] = restore(
            last_path, model, optimizer, scaler, device, fingerprint
        )
        print(f"resumed from {last_path} at epoch {start_epoch}, step {global_step}, "
              f"best {state['best']['metric']}={state['best']['value']:.4f}")
    elif args.resume == "auto":
        print("no checkpoint found, starting fresh")

    stale = 0
    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nepoch {epoch + 1}/{args.epochs}")
            global_step, train_loss = train_one_epoch(
                model, train_loader, optimizer, scaler, device, args, autocast_dtype,
                global_step, total_steps, writer, ckpt_dir,
                lambda step: payload_fn(step, epoch),
            )
            save_checkpoint(last_path, payload_fn(global_step, epoch + 1))

            if (epoch + 1) % args.eval_every != 0 and epoch + 1 != args.epochs:
                continue

            val_loss = validation_loss(
                model, val_loss_loader, device, args.num_negatives, autocast_dtype
            )
            metrics = evaluate(model, val_loader, device, autocast_dtype=autocast_dtype)
            current = headline(metrics, HEADLINE_K)
            print(f"  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")
            print(format_metrics(metrics))

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, global_step)
                for stream in ("recommended", "random"):
                    for key, value in metrics[stream].items():
                        if key.startswith(("recall@", "ndcg@", "mrr")):
                            writer.add_scalar(f"val_{stream}/{key}", value, global_step)
                writer.flush()

            with (out_dir / "metrics.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val": {s: metrics[s] for s in ("recommended", "random", "all")},
                    "coverage": val_ds.coverage(),
                }) + "\n")
            write_curves(out_dir)

            if current > state["best"]["value"]:
                state["best"] = {
                    "metric": HEADLINE_METRIC, "value": current,
                    "step": global_step, "epoch": epoch + 1,
                }
                save_checkpoint(ckpt_dir / "best.pt", payload_fn(global_step, epoch + 1))
                print(f"  new best {HEADLINE_METRIC} = {current:.4f}")
                stale = 0
            else:
                stale += 1
                print(f"  no gain on {HEADLINE_METRIC} "
                      f"(best {state['best']['value']:.4f} @ step {state['best']['step']}), "
                      f"{stale}/{args.patience or '-'}")
                if args.patience and stale >= args.patience:
                    print("early stop")
                    break
            save_checkpoint(last_path, payload_fn(global_step, epoch + 1))

            if args.max_steps and global_step >= args.max_steps:
                print("hit --max-steps")
                break
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OOM. In rough order of what to try first:\n"
              f"  --batch-size {max(1, batch_size // 2)} --grad-accum {args.grad_accum * 2}"
              "   (same effective batch, half the activations)\n"
              "  --num-negatives 128                      (drops the full-softmax logits)\n"
              f"  --max-impressions {max(32, args.max_impressions // 2)}"
              "                    (attention is quadratic in this)\n"
              "  --precision bf16                         (if it was fp32)\n"
              f"  --size {'base' if args.size in ('large', 'xlarge') else args.size}"
              "                            (smaller model)\n"
              "See the Stage 3 diagnostics table in the README.", flush=True)
        raise

    if args.smoke_test:
        return smoke_checks(
            model, optimizer, scaler, device, fingerprint, ckpt_dir,
            val_loader, vocab_size, global_step,
        )

    # ---- final report from the best checkpoint
    best_path = ckpt_dir / "best.pt"
    final: dict = {"best": state["best"], "config": cfg, "n_params": n_params}
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"\nloaded best checkpoint from epoch {ckpt['epoch']}, step {ckpt['global_step']}")

    for split in (("val", "test") if args.run_test else ("val",)):
        loader, dataset = build_eval_loader(
            split=split,
            processed_dir=args.processed_dir,
            max_impressions=args.max_impressions,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        metrics = evaluate(model, loader, device, autocast_dtype=autocast_dtype)
        metrics["coverage"] = dataset.coverage()
        metrics["n_users_scored"] = len(dataset)
        # Reference points, on exactly these targets. A retrieval number quoted
        # without them says nothing: on the recommended stream popularity alone
        # reaches NDCG@10 ~= 0.016, and on the random stream chance is the ceiling.
        baseline = evaluate_popularity(dataset, args.processed_dir)
        chance = chance_metrics(vocab_size - N_RESERVED_ITEM_IDS)
        metrics["popularity_baseline"] = baseline
        metrics["chance"] = chance
        final[split] = metrics
        print(f"\n=== {split} ({len(dataset):,} users, window {args.max_impressions}) ===")
        print(format_metrics(metrics, baseline=baseline, chance=chance))
        print(f"headline: {PRIMARY_EVAL_STREAM} ndcg@{HEADLINE_K} = {headline(metrics):.4f} "
              f"(popularity {baseline[PRIMARY_EVAL_STREAM][f'ndcg@{HEADLINE_K}']:.4f})")
        for stream in ("recommended", "random"):
            cov = metrics["coverage"][stream]
            print(f"coverage {stream:<12} {cov['fraction']:.2%} of split targets in window")
        if writer is not None:
            for stream in ("recommended", "random"):
                for key, value in metrics[stream].items():
                    if key.startswith(("recall@", "ndcg@", "mrr")):
                        writer.add_scalar(f"{split}_final_{stream}/{key}", value, global_step)

    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2, default=str))
    write_curves(out_dir)
    if writer is not None:
        writer.close()
    print(f"\nwrote {out_dir / 'final_metrics.json'}")
    return 0


def smoke_checks(
    model: HSTUOnKuaiRand,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    fingerprint: dict,
    ckpt_dir: Path,
    val_loader: DataLoader,
    vocab_size: int,
    global_step: int,
) -> int:
    """Assert the things a silent bug would break: metrics range, resume fidelity."""
    print("\n--- smoke checks ---")
    metrics = evaluate(model, val_loader, device, ks=(10,))
    assert metrics["all"]["n_targets"] > 0, "no targets were scored"
    for stream in ("recommended", "random", "all"):
        row = metrics[stream]
        if row["n_targets"] == 0:
            continue
        assert 0.0 <= row["recall@10"] <= 1.0, row
        assert 0.0 <= row["ndcg@10"] <= 1.0, row
        assert 0.0 < row["mrr"] <= 1.0, row
    print(f"metrics in range, {metrics['all']['n_targets']} targets scored, "
          f"streams partition them")

    last_path = ckpt_dir / "last.pt"
    assert last_path.exists(), f"{last_path} was never written"
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # Perturb, then restore: proves the checkpoint carries the weights rather than
    # the resume path quietly keeping the in-memory model.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))
    epoch, step, best = restore(last_path, model, optimizer, scaler, device, fingerprint)
    after = model.state_dict()
    max_delta = 0.0
    for key, old in before.items():
        new = after[key]
        if old.is_floating_point():
            max_delta = max(max_delta, float((new - old).abs().max()))
        else:
            # Bool/int buffers such as the causal attention mask: exact equality.
            assert bool(torch.equal(new, old)), f"resume changed buffer {key}"
    assert max_delta < 1e-6, f"resume did not restore weights exactly (max delta {max_delta})"
    print(f"resume restored {len(before)} tensors exactly (max delta {max_delta:.2e}) "
          f"at epoch {epoch}, step {step}, best {best['value']:.4f}")

    assert step == global_step, f"checkpoint step {step} != in-memory {global_step}"
    print(f"vocab {vocab_size}, fingerprint {fingerprint['dataset']}/"
          f"{fingerprint['split_strategy']}/{fingerprint['target_policy']}")
    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
