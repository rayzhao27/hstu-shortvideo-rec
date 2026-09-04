"""KuaiRand sequences -> the tensors Meta's HSTU encoder consumes.

Does not read MovieLens, Amazon, or any file under third_party/. The only inputs
are Stage 1 ``*_seqs.pkl`` plus the item encoder (for vocab size).

Each impression stays one row. Interleaving item/action tokens happens in the
model, not here: that way the protocol mask (which positions are recommended
targets) stays aligned with the Stage 1 arrays.

Padding is on the right, chronological order (oldest → newest), left-truncated
to ``max_impressions``. After a left truncate the new position 0 is unmarked as
a target — there is no prefix inside the window, same rule Stage 1 uses.

Left-truncating drops targets, which for an *evaluation* split silently changes
the denominator of every metric. So the dataset counts what it dropped and
exposes :meth:`KuaiRandSequenceDataset.coverage`; ``models/evaluate.py`` writes
that number next to the metrics instead of reporting a mean over an unstated
population. At ``max_impressions=256`` coverage is >=99.6% on every split and
stream; at 64 it falls to 89.7% on test/random.

``is_rand`` rides along on every row so one pass over the data can be scored on
both evaluation streams. Selecting a stream is a mask on the batch, which is the
same rule :mod:`data.protocol` applies at read time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.encoders import IdEncoder
from data.loader import PROCESSED_DIR
from data.protocol import EVAL_STREAMS, TRAIN_STREAM, target_mask
from data.sequences import load_sequences


class KuaiRandSequenceDataset(Dataset):
    """One Stage 1 split, already filtered to users who have at least one target."""

    def __init__(
        self,
        records: list,
        max_impressions: int,
        stream: str = TRAIN_STREAM,
        require_target: bool = True,
        limit_users: int | None = None,
    ) -> None:
        if max_impressions < 2:
            raise ValueError("max_impressions must be at least 2 (one history + one target)")
        self.max_impressions = int(max_impressions)
        self.stream = stream
        self.records: list = []
        self._targets_before = dict.fromkeys(("all",) + EVAL_STREAMS, 0)
        self._targets_after = dict.fromkeys(("all",) + EVAL_STREAMS, 0)
        for record in records:
            packed = self._prepare(record)
            if packed is None:
                continue
            if require_target and int(packed["supervision"].sum()) == 0:
                continue
            self.records.append(packed)
            if limit_users is not None and len(self.records) >= limit_users:
                break

    @classmethod
    def from_processed(
        cls,
        split: str = "train",
        processed_dir: Path = PROCESSED_DIR,
        max_impressions: int = 64,
        stream: str = TRAIN_STREAM,
        require_target: bool = True,
        limit_users: int | None = None,
    ) -> "KuaiRandSequenceDataset":
        path = Path(processed_dir) / f"{split}_seqs.pkl"
        return cls(
            load_sequences(path),
            max_impressions=max_impressions,
            stream=stream,
            require_target=require_target,
            limit_users=limit_users,
        )

    def _prepare(self, record: dict) -> dict | None:
        items = np.asarray(record["items"], dtype=np.int64)
        n = int(items.size)
        if n < 2:
            return None
        actions = np.asarray(record["actions"], dtype=np.int64)
        timestamps = np.asarray(record["timestamps"], dtype=np.int64)
        is_rand = np.asarray(record["is_rand"], dtype=np.int64)
        supervision = target_mask(record, self.stream).astype(np.int64)
        self._count(self._targets_before, supervision, is_rand)

        if n > self.max_impressions:
            items = items[-self.max_impressions :]
            actions = actions[-self.max_impressions :]
            timestamps = timestamps[-self.max_impressions :]
            is_rand = is_rand[-self.max_impressions :]
            supervision = supervision[-self.max_impressions :]
        supervision[0] = 0
        self._count(self._targets_after, supervision, is_rand)

        length = int(items.size)
        return {
            "user": int(record["user"]),
            "length": length,
            "items": items,
            "actions": actions,
            "timestamps": timestamps,
            "is_rand": is_rand,
            "supervision": supervision,
        }

    @staticmethod
    def _count(into: dict, supervision: np.ndarray, is_rand: np.ndarray) -> None:
        marked = supervision.astype(bool)
        into["all"] += int(marked.sum())
        into["recommended"] += int((marked & (is_rand == 0)).sum())
        into["random"] += int((marked & (is_rand == 1)).sum())

    def coverage(self) -> dict:
        """Targets kept inside the window vs targets the split actually holds.

        A metric averaged over ``kept`` is only a metric over the whole split if
        ``fraction`` is ~1.0. Reported, not assumed.
        """
        out = {}
        for stream, before in self._targets_before.items():
            after = self._targets_after[stream]
            out[stream] = {
                "targets_in_split": before,
                "targets_in_window": after,
                "fraction": (after / before) if before else 0.0,
            }
        return out

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


def collate_kuairand(batch: list, max_impressions: int) -> dict:
    """Right-pad a list of prepared records to ``max_impressions``."""
    bsz = len(batch)
    items = torch.zeros((bsz, max_impressions), dtype=torch.int64)
    actions = torch.zeros((bsz, max_impressions), dtype=torch.int64)
    timestamps = torch.zeros((bsz, max_impressions), dtype=torch.int64)
    supervision = torch.zeros((bsz, max_impressions), dtype=torch.float32)
    # -1 on padding so a padded slot can never be mistaken for either stream.
    is_rand = torch.full((bsz, max_impressions), -1, dtype=torch.int64)
    lengths = torch.zeros((bsz,), dtype=torch.int64)
    users = torch.zeros((bsz,), dtype=torch.int64)

    for i, row in enumerate(batch):
        n = int(row["length"])
        items[i, :n] = torch.from_numpy(row["items"])
        actions[i, :n] = torch.from_numpy(row["actions"])
        timestamps[i, :n] = torch.from_numpy(row["timestamps"])
        is_rand[i, :n] = torch.from_numpy(row["is_rand"])
        supervision[i, :n] = torch.from_numpy(row["supervision"].astype(np.float32))
        lengths[i] = n
        users[i] = int(row["user"])

    return {
        "users": users,
        "past_lengths": lengths,
        "past_ids": items,
        "actions": actions,
        "timestamps": timestamps,
        "is_rand": is_rand,
        "supervision": supervision,
    }


def load_item_vocab(processed_dir: Path = PROCESSED_DIR) -> int:
    """Embedding-table width: PAD/UNK included."""
    encoder = IdEncoder.load(Path(processed_dir) / "item_encoder.pkl")
    return int(encoder.vocab_size)
