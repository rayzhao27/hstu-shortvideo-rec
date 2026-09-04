"""Locate, clone, and import Meta's generative-recommenders.

The official package is not vendored. Stage 2 clones it into third_party/ (gitignored)
and puts that directory on sys.path so ``import generative_recommenders`` resolves.

HSTU's public encoder calls three ``torch.ops.fbgemm`` kernels (cumsum, dense↔jagged).
Those ship with ``fbgemm_gpu`` on the Ubuntu/CUDA stack the official README names.
On a Mac or a Colab runtime where the wheel does not match, we register a dense
PyTorch fallback with the same signatures so the research HSTU still runs. The
fallback is correct, just not the fast kernel. Training on A100 should use the
native ops when they import.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "third_party" / "generative-recommenders"
META_URL = "https://github.com/meta-recsys/generative-recommenders.git"

# Files the smoke test / README point at. Paths are relative to META_DIR.
HSTU_ENCODER = "generative_recommenders/research/modeling/sequential/hstu.py"
MFALCON_PATH = (
    "generative_recommenders/research/modeling/sequential/hstu.py"
    "  (HSTU.encode cache / delta_x_offsets — M-FALCON is not a separate module "
    "in the public release)"
)
SAMPLED_SOFTMAX = "generative_recommenders/research/modeling/sequential/losses/sampled_softmax.py"
TRAIN_LOOP = "generative_recommenders/research/trainer/train.py"


def clone_meta(force: bool = False) -> Path:
    """Clone the official repo at depth 1 if it is missing."""
    marker = META_DIR / "generative_recommenders" / "research" / "modeling" / "sequential" / "hstu.py"
    if marker.is_file() and not force:
        return META_DIR
    META_DIR.parent.mkdir(parents=True, exist_ok=True)
    if META_DIR.exists() and force:
        raise RuntimeError(f"{META_DIR} exists; delete it before --force")
    if META_DIR.exists() and not marker.is_file():
        raise RuntimeError(f"{META_DIR} exists but looks incomplete; delete it and retry")
    subprocess.run(
        ["git", "clone", "--depth", "1", META_URL, str(META_DIR)],
        check=True,
    )
    if not marker.is_file():
        raise RuntimeError(f"clone succeeded but {marker} is missing")
    return META_DIR


def add_to_sys_path() -> Path:
    """Put the official package root on sys.path. Idempotent."""
    path = str(META_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    return META_DIR


def _native_fbgemm_works() -> bool:
    try:
        import fbgemm_gpu  # noqa: F401
        import torch

        torch.ops.fbgemm.asynchronous_complete_cumsum(torch.zeros(2, dtype=torch.int64))
        return True
    except Exception:
        return False


class _FbgemmFallback:
    """Dense stand-ins for the three fbgemm ops the research HSTU actually calls."""

    @staticmethod
    def asynchronous_complete_cumsum(x):
        import torch

        x = x.to(torch.int64)
        return torch.cat([x.new_zeros(1), x.cumsum(0)], dim=0)

    @staticmethod
    def dense_to_jagged(dense, offsets):
        import torch

        off = offsets[0].to(device=dense.device, dtype=torch.int64)
        batch, width = dense.shape[0], dense.shape[1]
        lengths = off[1:] - off[:-1]
        if bool((lengths < 0).any()) or bool((lengths > width).any()):
            raise ValueError(f"dense_to_jagged lengths {lengths.tolist()} incompatible with width {width}")
        positions = torch.arange(width, device=dense.device).unsqueeze(0).expand(batch, width)
        mask = positions < lengths.unsqueeze(1)
        return (dense[mask],)

    @staticmethod
    def jagged_to_padded_dense(values, offsets, max_lengths, padding_value=0.0):
        import torch

        off = offsets[0].to(device=values.device, dtype=torch.int64)
        max_len = int(max_lengths[0] if not torch.is_tensor(max_lengths[0]) else max_lengths[0].item())
        batch = int(off.numel() - 1)
        tail = tuple(values.shape[1:])
        out = values.new_full((batch, max_len) + tail, padding_value)
        lengths = off[1:] - off[:-1]
        if values.numel() == 0 or int(lengths.sum()) == 0:
            return out
        batch_idx = torch.repeat_interleave(torch.arange(batch, device=values.device), lengths)
        starts = torch.repeat_interleave(off[:-1], lengths)
        pos_idx = torch.arange(values.size(0), device=values.device) - starts
        out[batch_idx, pos_idx] = values
        return out


def install_fbgemm_fallback() -> str:
    """Use native fbgemm_gpu when it works; otherwise bind a dense fallback.

    Returns ``\"native\"`` or ``\"fallback\"``.
    """
    import torch

    if _native_fbgemm_works():
        return "native"

    fallback = _FbgemmFallback()
    # torch.ops.fbgemm is normally created by loading the C++ library. When that
    # library is absent, getattr raises. Assigning a simple object works because
    # HSTU looks the ops up at call time as torch.ops.fbgemm.<name>(...).
    try:
        object.__setattr__(torch.ops, "fbgemm", fallback)
    except (AttributeError, TypeError):
        torch.ops.__dict__["fbgemm"] = fallback
    return "fallback"


def prepare(clone: bool = True) -> dict:
    """Clone if needed, put the package on sys.path, bind fbgemm. Safe to call twice."""
    if clone:
        clone_meta()
    elif not (META_DIR / "generative_recommenders").is_dir():
        raise FileNotFoundError(
            f"{META_DIR} is missing. Run python -m utils.meta_repo or pass clone=True."
        )
    add_to_sys_path()
    backend = install_fbgemm_fallback()
    return {
        "meta_dir": str(META_DIR),
        "fbgemm": backend,
        "hstu_encoder": HSTU_ENCODER,
        "m_falcon": MFALCON_PATH,
        "loss": SAMPLED_SOFTMAX,
        "train_loop": TRAIN_LOOP,
    }


def main(argv: list | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Clone Meta generative-recommenders.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = clone_meta(force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
