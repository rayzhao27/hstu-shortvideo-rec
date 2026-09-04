"""Remap raw KuaiRand ids to contiguous integers usable as embedding rows.

Index 0 is PAD and index 1 is UNK, so real ids start at 2. Both slots exist for
Stage 2: padded batches need a row that means "nothing here", and an item that the
training split never saw needs a row that means "never seen".

The item encoder is fit on the **train split only**. Fitting it on the whole dataset
would put items into the vocabulary that the model could not have known about at
training time, which quietly inflates evaluation. On KuaiRand-Pure the cost of doing
it properly is tiny - only 3 videos appear exclusively in the last 7 days - but the
OOV rate is reported per split so the number is visible rather than assumed.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PAD_IDX = 0
UNK_IDX = 1
N_RESERVED = 2


@dataclass
class IdEncoder:
    """Bidirectional map between raw ids and contiguous indices."""

    name: str
    raw_to_idx: dict
    idx_to_raw: np.ndarray  # position i holds the raw id of index i (-1 for PAD/UNK)

    @classmethod
    def fit(cls, values: np.ndarray, name: str) -> IdEncoder:
        uniques = np.unique(np.asarray(values))
        raw_to_idx = {int(raw): i + N_RESERVED for i, raw in enumerate(uniques)}
        idx_to_raw = np.concatenate([np.full(N_RESERVED, -1, dtype=np.int64), uniques.astype(np.int64)])
        logger.info("%s encoder: %d ids -> indices %d..%d",
                    name, len(uniques), N_RESERVED, N_RESERVED + len(uniques) - 1)
        return cls(name=name, raw_to_idx=raw_to_idx, idx_to_raw=idx_to_raw)

    @property
    def vocab_size(self) -> int:
        """Number of embedding rows, PAD and UNK included."""
        return len(self.idx_to_raw)

    @property
    def n_known(self) -> int:
        return len(self.raw_to_idx)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Map raw ids to indices, sending anything unseen to UNK."""
        lookup = self.raw_to_idx
        return np.fromiter(
            (lookup.get(int(v), UNK_IDX) for v in np.asarray(values)),
            dtype=np.int32,
            count=len(values),
        )

    def oov_mask(self, values: np.ndarray) -> np.ndarray:
        lookup = self.raw_to_idx
        return np.fromiter(
            (int(v) not in lookup for v in np.asarray(values)), dtype=bool, count=len(values)
        )

    def inverse_transform(self, indices: np.ndarray) -> np.ndarray:
        return self.idx_to_raw[np.asarray(indices)]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("wrote %s (%d entries)", path, self.n_known)
        return path

    @staticmethod
    def load(path: Path) -> IdEncoder:
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_known_ids": self.n_known,
            "vocab_size": self.vocab_size,
            "pad_idx": PAD_IDX,
            "unk_idx": UNK_IDX,
            "first_real_idx": N_RESERVED,
        }
