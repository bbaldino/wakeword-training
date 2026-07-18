"""Deterministic train/eval partition of collected wake clips.

The whole point of the flywheel is to know a model actually improved. That only
works if evaluation measures *generalization*, not memorization — so a slice of
the collected clips must be held out from training and used only for scoring.

A clip's bucket is a stable hash of its id, so the same clip always lands in the
same bucket across runs and across both consumers: ``pull_negatives.py`` trains
on the TRAIN clips only, ``evaluate.py`` scores on the EVAL clips only. No clip
is ever both trained on and tested on.

Env:
  EVAL_FRACTION   fraction of clips reserved for evaluation (default 0.2)
"""
import hashlib
import os

EVAL_FRACTION = float(os.environ.get("EVAL_FRACTION", "0.2"))
_BUCKETS = 1000


def is_eval_clip(clip_id: str) -> bool:
    """True if this clip belongs to the held-out evaluation set."""
    h = int(hashlib.sha256(clip_id.encode()).hexdigest(), 16) % _BUCKETS
    return h < EVAL_FRACTION * _BUCKETS
