"""Report the runtime: python, torch, GPU. Stage 0 step 1.

    python -m utils.check_env
"""

from __future__ import annotations

import os
import platform

# Ampere or newer, where bf16 and FlashAttention-style kernels are available.
PREFERRED_GPUS = ("A100", "H100", "H200", "L4", "L40", "A10")


def cpu_ram_gib() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        return float("nan")


def main() -> int:
    print(f"python   {platform.python_version()}")
    print(f"platform {platform.platform()}")
    print(f"cpu ram  {cpu_ram_gib():.1f} GiB")

    try:
        import torch
    except ImportError:
        print("torch    not installed")
        print("WARN     install torch before the training stages")
        return 0

    print(f"torch    {torch.__version__} (cuda {torch.version.cuda})")

    if not torch.cuda.is_available():
        print("gpu      none")
        print("WARN     no GPU visible; in Colab use Runtime > Change runtime type > GPU")
        return 0

    props = torch.cuda.get_device_properties(0)
    print(
        f"gpu      {props.name} | {props.total_memory / 1024**3:.1f} GiB "
        f"| sm {props.major}.{props.minor} | {props.multi_processor_count} SMs"
    )
    print(f"bf16     {torch.cuda.is_bf16_supported()}")

    if any(key in props.name.upper() for key in PREFERRED_GPUS):
        print(f"status   OK, {props.name} is what Stage 0 asks for")
    else:
        print(
            f"status   USABLE but not preferred. {props.name} is not in {PREFERRED_GPUS}; "
            "halve max_seq_len and batch_size, and use fp16 if bf16 is unsupported"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
