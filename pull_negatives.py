"""Pull the puck's hard negatives for one wake word and turn them into
openWakeWord training features.

Negatives for model X are the reviewed clips where X was NOT said — a different
wake word, or no wake word at all (GET {ORCHESTRATOR_URL}/events/export?not_said=X;
see puck_clips.py for why this selects by what was said, not by `label`).
Featurizes them with openWakeWord's AudioFeatures.embed_clips, saves a negatives
.npy, and patches the training config so `--train_model` folds them in alongside
the synthetic negatives.

Run in the training container after config generation (train.sh Step 2), before
the train step. Skips cleanly if ORCHESTRATOR_URL is unset or there are no clips.

Env:
  ORCHESTRATOR_URL  puck API base, e.g. http://192.168.1.145  (required)
  MODEL_NAME        model being trained, e.g. hey_tars        (required)
  NEG_OUTPUT        output .npy         (default /data/custom_negatives.npy)
  NEG_CONFIG        config to patch     (default /app/config.yaml)
  NEG_BATCH_N       batch_n_per_class weight for the negatives (default 50)
  NEG_CLIP_SECS     pad/trim clips to this length (default 2.0)
"""
import os

import numpy as np
import yaml
from openwakeword.utils import AudioFeatures

from eval_split import is_eval_clip
from puck_clips import fetch_clips as fetch_puck_clips

RATE = 16000
ORCH = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
NAME = os.environ.get("MODEL_NAME", "")
OUT = os.environ.get("NEG_OUTPUT", "/data/custom_negatives.npy")
CFG = os.environ.get("NEG_CONFIG", "/app/config.yaml")
BATCH_N = int(os.environ.get("NEG_BATCH_N", "50"))
CLIP_LEN = int(float(os.environ.get("NEG_CLIP_SECS", "2.0")) * RATE)


def fetch_clips() -> np.ndarray:
    print(f"Pulling clips where {NAME!r} was not said from {ORCH}")
    clips, held_out = [], 0
    for clip_id, a in fetch_puck_clips(ORCH, CLIP_LEN, not_said=NAME):
        # Reserve the held-out eval slice so evaluate.py measures
        # generalization, not clips the model was trained on.
        if is_eval_clip(clip_id):
            held_out += 1
            continue
        clips.append(a)
    if held_out:
        print(f"Held out {held_out} clip(s) for evaluation (not used for training)")
    return np.stack(clips).astype(np.int16) if clips else np.empty((0, CLIP_LEN), np.int16)


def main() -> None:
    if not ORCH or not NAME:
        print("ORCHESTRATOR_URL and MODEL_NAME required; skipping custom negatives.")
        return
    clips = fetch_clips()
    print(f"Got {len(clips)} negative clips")
    if len(clips) == 0:
        print("No negative clips; skipping.")
        return

    feats = AudioFeatures().embed_clips(clips, batch_size=64)
    np.save(OUT, feats)
    print(f"Saved negative features {feats.shape} -> {OUT}")

    if os.path.exists(CFG):
        with open(CFG) as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("feature_data_files", {})["custom_negatives"] = OUT
        cfg.setdefault("batch_n_per_class", {})["custom_negatives"] = BATCH_N
        with open(CFG, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"Patched {CFG}: custom_negatives (batch_n_per_class={BATCH_N})")
    else:
        print(f"NOTE: {CFG} not found. Add manually:")
        print(f'  feature_data_files["custom_negatives"]: "{OUT}"')
        print(f'  batch_n_per_class["custom_negatives"]: {BATCH_N}')


if __name__ == "__main__":
    main()
