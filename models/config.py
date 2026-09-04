"""Model size ladder.

One place to name shapes, so Stage 3 (train one model) and Stage 5 (scaling law
across sizes) cannot drift apart. Stage 5 reuses these presets by name.

The ladder follows Meta's own recipe: per-head dim is held at 32 and capacity grows
through ``embedding_dim`` and ``num_blocks``. Their published ml-20m HSTU-large is
``embedding_dim=256, num_blocks=16, num_heads=8, dqk=dv=32``
(configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin), so ``large`` here is one
rung below it and ``xlarge`` is roughly it.

``suggested_batch_size`` is what fits an L4 (24GB) at ``max_impressions=256`` (a
513-token axis) with full-softmax loss over the ~7.6k KuaiRand-Pure catalogue. An
A100 40GB takes roughly 2x that. These are starting points, not measurements of your
runtime - see the OOM table in the README.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSize:
    """A named HSTU shape. ``None`` per-head dims mean embedding_dim // num_heads."""

    name: str
    embedding_dim: int
    num_blocks: int
    num_heads: int
    attention_dim: int | None = None
    linear_dim: int | None = None
    suggested_batch_size: int = 64
    note: str = ""

    @property
    def dqk(self) -> int:
        return self.attention_dim or self.embedding_dim // self.num_heads

    @property
    def dv(self) -> int:
        return self.linear_dim or self.embedding_dim // self.num_heads

    def approx_params(self, vocab_size: int, max_impressions: int, n_actions: int = 7) -> int:
        """Parameter count without building the model, for the scaling-law axis."""
        emb = vocab_size * self.embedding_dim
        emb += n_actions * self.embedding_dim
        emb += (max_impressions * 2) * self.embedding_dim
        per_block = self.embedding_dim * (self.dv * 2 * self.num_heads + self.dqk * 2 * self.num_heads)
        per_block += self.dv * self.num_heads * self.embedding_dim + self.embedding_dim
        return emb + self.num_blocks * per_block

    def to_dict(self) -> dict:
        out = asdict(self)
        out["dqk"] = self.dqk
        out["dv"] = self.dv
        return out


SIZES: dict[str, ModelSize] = {
    "tiny": ModelSize(
        name="tiny",
        embedding_dim=32,
        num_blocks=2,
        num_heads=1,
        suggested_batch_size=32,
        note="smoke tests and CPU runs only, not a research setting",
    ),
    "small": ModelSize(
        name="small",
        embedding_dim=64,
        num_blocks=2,
        num_heads=2,
        suggested_batch_size=128,
        note="lowest rung of the scaling ladder",
    ),
    "base": ModelSize(
        name="base",
        embedding_dim=128,
        num_blocks=4,
        num_heads=4,
        suggested_batch_size=64,
        note="Stage 3 default; fits an L4 with room to spare",
    ),
    "large": ModelSize(
        name="large",
        embedding_dim=256,
        num_blocks=8,
        num_heads=8,
        suggested_batch_size=32,
        note="A100 recommended; on an L4 drop batch size or max_impressions",
    ),
    "xlarge": ModelSize(
        name="xlarge",
        embedding_dim=256,
        num_blocks=16,
        num_heads=8,
        suggested_batch_size=16,
        note="A100 only; mirrors Meta's published HSTU-large shape",
    ),
}

DEFAULT_SIZE = "base"
SMOKE_SIZE = "tiny"

# Held fixed across the ladder so Stage 5 varies capacity and nothing else.
SCALING_LADDER: tuple[str, ...] = ("small", "base", "large", "xlarge")


def get_size(name: str) -> ModelSize:
    if name not in SIZES:
        raise ValueError(f"unknown size {name!r}, expected one of {sorted(SIZES)}")
    return SIZES[name]


def describe_ladder(vocab_size: int = 7582, max_impressions: int = 256) -> str:
    rows = [
        f"{'size':<8}{'dim':>5}{'blocks':>8}{'heads':>7}{'dqk=dv':>8}"
        f"{'~params':>12}{'batch':>7}  note"
    ]
    for name, size in SIZES.items():
        rows.append(
            f"{name:<8}{size.embedding_dim:>5}{size.num_blocks:>8}{size.num_heads:>7}"
            f"{size.dqk:>8}{size.approx_params(vocab_size, max_impressions):>12,}"
            f"{size.suggested_batch_size:>7}  {size.note}"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe_ladder())
