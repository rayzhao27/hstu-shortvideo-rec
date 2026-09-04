"""Official HSTU encoder, wired to KuaiRand item + 7-class action sequences.

Interleaving matches Meta's CombinedItemAndRating preprocessor (item_0, action_0,
item_1, action_1, ...) but reads ``actions`` instead of MovieLens ratings. Each
action is one of the seven Stage 1 classes (PAD=0 plus SKIP..HATE).

Relative time is the official ``RelativeBucketedTimeAndPositionBasedBias``:
HSTU reads ``past_payloads[\"timestamps\"]`` and bucketizes
``log(|t_j - t_i|)``. Because interleaving doubles the token axis, each
impression timestamp is repeated on the item token and its action token — no
time passes between seeing a video and the feedback on it.

M-FALCON is the cached incremental path on ``HSTU.encode`` (``delta_x_offsets`` /
``HSTUCacheState``). The public repo does not ship a separately named module.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from data.actions import N_ACTIONS
from data.encoders import PAD_IDX, UNK_IDX
from utils.meta_repo import prepare

# Item ids 0 and 1 are PAD and UNK (data/encoders.py). Neither is a video, so
# neither may be sampled as a negative or returned as a recommendation.
N_RESERVED_ITEM_IDS = max(PAD_IDX, UNK_IDX) + 1

prepare()

from generative_recommenders.research.modeling.initialization import truncated_normal
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.hstu import TIMESTAMPS_KEY, HSTU
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
)
from generative_recommenders.research.rails.similarities.dot_product_similarity_fn import (
    DotProductSimilarity,
)


class CombinedItemAndActionPreprocessor(InputFeaturesPreprocessorModule):
    """[item_0, action_0, item_1, action_1, ...] with a shared embedding dim."""

    def __init__(
        self,
        max_impressions: int,
        embedding_dim: int,
        dropout_rate: float,
        num_actions: int = N_ACTIONS,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._max_tokens = max_impressions * 2
        self._pos_emb = torch.nn.Embedding(self._max_tokens, embedding_dim)
        self._action_emb = torch.nn.Embedding(num_actions, embedding_dim, padding_idx=0)
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._dropout_rate = dropout_rate
        self.reset_state()

    def debug_str(self) -> str:
        return f"combia_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._action_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        self._action_emb.weight.data[0].zero_()

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_imp = past_ids.size()
        dim = past_embeddings.size(-1)
        stacked = torch.stack(
            [past_embeddings, self._action_emb(past_payloads["actions"].long())],
            dim=2,
        ) * (self._embedding_dim ** 0.5)
        tokens = stacked.reshape(batch, n_imp * 2, dim)
        tokens = tokens + self._pos_emb(
            torch.arange(n_imp * 2, device=past_ids.device).unsqueeze(0).expand(batch, -1)
        )
        tokens = self._emb_dropout(tokens)
        valid = (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(batch, n_imp * 2)
        tokens = tokens * valid.unsqueeze(2).to(tokens.dtype)
        return past_lengths * 2, tokens, valid.unsqueeze(2).to(tokens.dtype)


def interleave_timestamps(timestamps: torch.Tensor) -> torch.Tensor:
    """Repeat each impression time on its item token and its action token."""
    return timestamps.unsqueeze(-1).expand(-1, -1, 2).reshape(timestamps.size(0), -1)


class HSTUOnKuaiRand(torch.nn.Module):
    """Official ``HSTU`` plus the KuaiRand item/action preprocessor."""

    def __init__(
        self,
        num_items: int,
        max_impressions: int,
        embedding_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        attention_dim: int | None = None,
        linear_dim: int | None = None,
        dropout_rate: float = 0.2,
        max_output_len: int = 1,
        enable_relative_attention_bias: bool = True,
        item_l2_norm: bool = True,
        temperature: float = 0.05,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.max_impressions = int(max_impressions)
        self.max_output_len = int(max_output_len)
        self.embedding_dim = int(embedding_dim)
        self.num_items = int(num_items)
        # Per-head dims, as in Meta's own configs: ml-20m HSTU-large is
        # embedding_dim=256 / num_heads=8 / dqk=dv=32. Holding dqk=dv=D/H keeps
        # per-block parameters at ~5*D^2 regardless of head count, which is what
        # makes the Stage 5 size ladder a clean capacity axis.
        self.attention_dim = int(attention_dim or embedding_dim // num_heads)
        self.linear_dim = int(linear_dim or embedding_dim // num_heads)
        self.item_l2_norm = bool(item_l2_norm)
        self.temperature = float(temperature)
        # LocalEmbeddingModule allocates num_items+1 rows with padding_idx=0.
        # Our encoder already reserves 0=PAD, so pass vocab_size-1.
        self.embedding = LocalEmbeddingModule(
            num_items=num_items - 1,
            item_embedding_dim=embedding_dim,
        )
        preprocessor = CombinedItemAndActionPreprocessor(
            max_impressions=max_impressions,
            embedding_dim=embedding_dim,
            dropout_rate=dropout_rate,
        )
        self.hstu = HSTU(
            max_sequence_len=max_impressions * 2,
            max_output_len=max_output_len,
            embedding_dim=embedding_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            linear_dim=self.linear_dim,
            attention_dim=self.attention_dim,
            normalization="rel_bias",
            linear_config="uvqk",
            linear_activation="silu",
            linear_dropout_rate=dropout_rate,
            attn_dropout_rate=dropout_rate,
            embedding_module=self.embedding,
            similarity_module=DotProductSimilarity(),
            input_features_preproc_module=preprocessor,
            output_postproc_module=L2NormEmbeddingPostprocessor(embedding_dim=embedding_dim),
            enable_relative_attention_bias=enable_relative_attention_bias,
            verbose=verbose,
        )

    @property
    def token_length(self) -> int:
        return self.max_impressions * 2

    def _payloads(self, batch: dict) -> dict:
        timestamps = interleave_timestamps(batch["timestamps"])
        need = self.token_length + self.max_output_len
        if timestamps.size(1) < need:
            timestamps = F.pad(timestamps, (0, need - timestamps.size(1)))
        elif timestamps.size(1) > need:
            timestamps = timestamps[:, :need]
        return {
            "actions": batch["actions"],
            TIMESTAMPS_KEY: timestamps,
        }

    def encode(self, batch: dict) -> torch.Tensor:
        """(B, token_length + max_output_len, D) hidden states after HSTU."""
        past_ids = batch["past_ids"]
        encoded = self.hstu(
            past_lengths=batch["past_lengths"],
            past_ids=past_ids,
            past_embeddings=self.hstu.get_item_embeddings(past_ids),
            past_payloads=self._payloads(batch),
        )
        return encoded

    def next_item_queries(self, encoded: torch.Tensor) -> torch.Tensor:
        """Hidden state after each action token, used to predict the *next* item.

        Interleaved layout: even = item, odd = action. After (item_t, action_t)
        we predict item_{t+1}, so the queries are encoded[:, 1, 3, 5, ..., 2N-3].
        """
        action_states = encoded[:, 1 : self.token_length : 2, :]
        return action_states[:, :-1, :]

    # ---- scoring head -------------------------------------------------------
    # Encoder output is already unit-norm (L2NormEmbeddingPostprocessor), so with
    # item_l2_norm the logit is a cosine similarity and the temperature sets how
    # peaked the softmax is. Meta's validated setting is l2_norm + temperature
    # 0.05; both are exposed so the choice stays visible rather than baked in.

    def item_table(self) -> torch.Tensor:
        weight = self.embedding._item_emb.weight
        if self.item_l2_norm:
            weight = F.normalize(weight, p=2.0, dim=-1, eps=1e-6)
        return weight

    def full_logits(self, queries: torch.Tensor) -> torch.Tensor:
        """(..., D) -> (..., vocab), unmasked.

        PAD and UNK are *not* set to -inf here. Ranking must exclude them, and
        ``models/evaluate.py`` does, but a -inf column in a training loss is a
        NaN generator: masked-out positions carry target id 0 (PAD), and
        ``inf * 0`` is NaN, so the mask would no longer mask anything.
        """
        return torch.matmul(queries, self.item_table().t()) / self.temperature

    def next_item_loss(
        self,
        batch: dict,
        encoded: torch.Tensor | None = None,
        num_negatives: int | None = None,
    ) -> torch.Tensor:
        """Next-item CE at the target positions the protocol marks.

        Positions are gathered before scoring rather than scored and then
        multiplied by a 0/1 weight. Same number, but nothing is ever computed on
        padding, so a non-finite value at a padded slot cannot leak into the
        mean, and the logit tensor shrinks to the marked positions.

        ``num_negatives=None`` uses the full catalogue, which is exact and
        affordable at KuaiRand-Pure's ~7.6k items. Passing an integer switches to
        sampled softmax (the official HSTU objective), which is what makes larger
        catalogues and larger batches fit - see the OOM table in the README.
        """
        if encoded is None:
            encoded = self.encode(batch)
        queries = self.next_item_queries(encoded)
        marked = batch["supervision"][:, 1:] > 0
        if not bool(marked.any()):
            return queries.sum() * 0.0
        flat_queries = queries[marked]
        flat_targets = batch["past_ids"][:, 1:][marked]

        if num_negatives is None:
            logits = self.full_logits(flat_queries)
        else:
            logits, flat_targets = self._sampled_logits(
                flat_queries, flat_targets, num_negatives
            )
        return F.cross_entropy(logits, flat_targets)

    def _sampled_logits(
        self, queries: torch.Tensor, targets: torch.Tensor, num_negatives: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform sampled softmax, negatives shared across the batch.

        Shared negatives keep the logit tensor at (T, 1+N) instead of the
        (T, N, D) gather a per-position sampler needs. Any negative colliding
        with the true item is masked out, otherwise the objective would push down
        the item it is simultaneously pushing up. The true item is column 0, so
        the labels are all zero.
        """
        table = self.item_table()
        negatives = torch.randint(
            N_RESERVED_ITEM_IDS,
            self.num_items,
            (num_negatives,),
            device=queries.device,
        )
        pos = (queries * table[targets]).sum(-1, keepdim=True) / self.temperature
        neg = torch.matmul(queries, table[negatives].t()) / self.temperature
        neg = neg.masked_fill(targets.unsqueeze(-1) == negatives.unsqueeze(0), float("-inf"))
        logits = torch.cat([pos, neg], dim=-1)
        labels = torch.zeros(queries.size(0), dtype=torch.int64, device=queries.device)
        return logits, labels
