"""Pull false-positive wake clips from the voice-orchestrator and turn them into
openWakeWord hard-negative training features.

Fetches GET {ORCHESTRATOR_URL}/events/export?label=false (a tar of 16 kHz mono
WAVs the puck flagged as false wakes), featurizes them with openWakeWord's
AudioFeatures.embed_clips, saves a negatives .npy, and patches the training
config so `--train_model` folds them in alongside the synthetic negatives.

Run in the training container after config generation (train.sh Step 2), before
the train step. Skips cleanly if ORCHESTRATOR_URL is unset or there are no clips.

Env:
  ORCHESTRATOR_URL  puck API base, e.g. http://192.168.1.145:8100  (required)
  NEG_OUTPUT        output .npy         (default /data/custom_negatives.npy)
  NEG_CONFIG        config to patch     (default /app/config.yaml)
  NEG_BATCH_N       batch_n_per_class weight for the negatives (default 50)
  NEG_CLIP_SECS     pad/trim clips to this length (default 2.0)
"""
import io
import os
import sys
import tarfile
import urllib.request
import wave

import numpy as np
import yaml
from openwakeword.utils import AudioFeatures

from eval_split import is_eval_clip

RATE = 16000
ORCH = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
OUT = os.environ.get("NEG_OUTPUT", "/data/custom_negatives.npy")
CFG = os.environ.get("NEG_CONFIG", "/app/config.yaml")
BATCH_N = int(os.environ.get("NEG_BATCH_N", "50"))
CLIP_LEN = int(float(os.environ.get("NEG_CLIP_SECS", "2.0")) * RATE)


def fetch_clips() -> np.ndarray:
    url = f"{ORCH}/events/export?label=false"
    print(f"Pulling false clips from {url}")
    data = urllib.request.urlopen(url, timeout=30).read()
    clips = []
    held_out = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".wav"):
                continue
            # Reserve the held-out eval slice so evaluate.py measures
            # generalization, not clips the model was trained on.
            clip_id = m.name.rsplit("/", 1)[-1][: -len(".wav")]
            if is_eval_clip(clip_id):
                held_out += 1
                continue
            with wave.open(tar.extractfile(m)) as w:
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            a = a[:CLIP_LEN] if len(a) >= CLIP_LEN else np.pad(a, (0, CLIP_LEN - len(a)))
            clips.append(a)
    if held_out:
        print(f"Held out {held_out} clip(s) for evaluation (not used for training)")
    return np.stack(clips).astype(np.int16) if clips else np.empty((0, CLIP_LEN), np.int16)


def main() -> None:
    if not ORCH:
        print("ORCHESTRATOR_URL not set; skipping custom negatives.")
        return
    clips = fetch_clips()
    print(f"Got {len(clips)} false-positive clips")
    if len(clips) == 0:
        print("No false clips; skipping.")
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
