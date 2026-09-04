"""Action taxonomy: turn KuaiRand's feedback flags into one discrete action per impression.

HSTU consumes a sequence of interleaved (item, action) tokens, so every impression
needs exactly one action id. KuaiRand instead gives ~8 independent binary flags that
overlap heavily, so we need an explicit, ordered collapse.

Design, and why:

* The positive flags form a near-perfect containment chain, measured on the standard
  log: 99.6% of ``long_view`` rows also have ``is_click``, 84% of ``is_like`` rows have
  ``is_click``, 43% of ``is_follow`` rows also have ``is_like``. So "strength of
  engagement" is a meaningful single axis and a priority ladder loses little.
* ``is_hate`` is *not* on that axis: only 45% of hated impressions were clicked and
  their median play ratio is 0.098. Ranking it as "very strong engagement" would be
  wrong, so it gets its own class at the top of the priority order (it is also the
  rarest signal at 0.05%, so it would otherwise be swallowed by the other classes).
* Author-directed signals (profile enter, follow, forward, comment) are grouped into
  one SOCIAL class. Individually they are 0.10%-2.4% of impressions, too rare to
  support their own embedding, and they express the same intent: interest in the
  creator rather than the single video.

The first matching rule wins, so the tuple order below *is* the priority order.
``SKIP`` matches everything and acts as the fallback, which guarantees total coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Reserved for padding in Stage 2; no impression ever gets this id.
ACTION_PAD = 0


@dataclass(frozen=True)
class ActionSpec:
    """One action class and the rule that recognises it."""

    id: int
    name: str
    description: str
    predicate: Callable[[pd.DataFrame], np.ndarray]

    def matches(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predicate(df), dtype=bool)


def _flag(df: pd.DataFrame, column: str) -> np.ndarray:
    """Read a binary flag, treating a missing column as all-zero.

    The 1K / 27K releases carry the same columns, but this keeps the encoder from
    exploding if a future release drops one.
    """
    if column not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df[column].to_numpy() == 1


def _any_flag(df: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    for column in columns:
        out |= _flag(df, column)
    return out


SOCIAL_FLAGS = ("is_follow", "is_forward", "is_comment", "is_profile_enter")

# Ordered by priority: the first rule that matches assigns the action.
ACTIONS: tuple[ActionSpec, ...] = (
    ActionSpec(
        6,
        "HATE",
        "explicit negative feedback (is_hate)",
        lambda df: _flag(df, "is_hate"),
    ),
    ActionSpec(
        5,
        "SOCIAL",
        "author-directed action: profile enter, follow, forward or comment",
        lambda df: _any_flag(df, SOCIAL_FLAGS),
    ),
    ActionSpec(
        4,
        "LIKE",
        "hit the like button (is_like)",
        lambda df: _flag(df, "is_like"),
    ),
    ActionSpec(
        3,
        "LONG_VIEW",
        "watched to the end if <=18s, else watched >=18s (long_view)",
        lambda df: _flag(df, "long_view"),
    ),
    ActionSpec(
        2,
        "CLICK",
        "clicked / valid play but not a long view (is_click)",
        lambda df: _flag(df, "is_click"),
    ),
    ActionSpec(
        1,
        "SKIP",
        "impression with no positive feedback, i.e. scrolled away",
        lambda df: np.ones(len(df), dtype=bool),
    ),
)

ACTION_IDS: tuple[int, ...] = tuple(spec.id for spec in ACTIONS)
ACTION_NAMES: dict[int, str] = {spec.id: spec.name for spec in ACTIONS}
ACTION_NAMES[ACTION_PAD] = "PAD"
N_ACTIONS = max(ACTION_IDS) + 1  # size of the action embedding table, PAD included


def encode_actions(df: pd.DataFrame) -> np.ndarray:
    """Map each row to exactly one action id, highest-priority rule first."""
    out = np.zeros(len(df), dtype=np.int8)
    for spec in ACTIONS:
        unassigned = out == ACTION_PAD
        if not unassigned.any():
            break
        out[spec.matches(df) & unassigned] = spec.id

    if (out == ACTION_PAD).any():
        raise ValueError("some impressions were not assigned an action")
    return out


def action_distribution(actions: np.ndarray) -> dict:
    """Count and share of every action class, keyed by action name."""
    total = int(actions.size)
    counts = np.bincount(actions, minlength=N_ACTIONS)
    return {
        ACTION_NAMES[action_id]: {
            "id": int(action_id),
            "count": int(counts[action_id]),
            "share": float(counts[action_id] / total) if total else 0.0,
        }
        for action_id in ACTION_IDS
    }


def describe_taxonomy() -> str:
    """Render the taxonomy for logs and for the README."""
    lines = ["action taxonomy (first matching rule wins):"]
    lines += [f"  {ACTION_PAD}  {'PAD':<10} reserved for padding, never assigned"]
    for spec in ACTIONS:
        lines.append(f"  {spec.id}  {spec.name:<10} {spec.description}")
    return "\n".join(lines)
