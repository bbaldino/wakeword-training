"""Offline evaluation gate: is the freshly-trained model actually better?

Scores a just-trained candidate model against the model currently deployed on
the puck, using data neither was allowed to train on:

  * held-out false wakes  (the eval slice of collected false clips) -> false
    accepts. This is the direct measure of whether feeding negatives helps: does
    the new model stop firing on adversarial clips it never saw in training?
  * confirmed real wakes  (label=real, from the puck)               -> misses
  * synthetic positives   (openWakeWord's own held-out test split)  -> misses
    (best effort; a quick baseline before real clips accumulate)

Everything is featurized with the same AudioFeatures pipeline training uses, so
the comparison is apples-to-apples. Report-only: it prints a verdict and writes
<MODEL>.eval.json next to the model. It does NOT block deployment.

Env:
  ORCHESTRATOR_URL   puck API base (required — supplies held-out clips + current model)
  MODEL_NAME         model stem, e.g. hey_tars (required)
  MODEL_PUSH_TOKEN   sent as X-Auth-Token if the puck is auth-gated (optional)
  THRESHOLD          operating detection threshold (default 0.5, matches the puck)
  OUTPUT_DIR         where the candidate .onnx lives (default /output)
  EVAL_SYNTH_GLOBS   comma globs for OWW held-out positive features
                     (default "*positive*test*.npy,*positive*val*.npy")
"""
import glob
import io
import json
import os
import sys
import tarfile
import urllib.request
import wave
from datetime import datetime

import numpy as np
import onnxruntime as ort
from openwakeword.utils import AudioFeatures

from eval_split import is_eval_clip

RATE = 16000
CLIP_LEN = int(float(os.environ.get("NEG_CLIP_SECS", "2.0")) * RATE)
ORCH = os.environ.get("ORCHESTRATOR_URL", "").rstrip("/")
NAME = os.environ.get("MODEL_NAME", "")
TOKEN = os.environ.get("MODEL_PUSH_TOKEN", "")
THRESHOLD = float(os.environ.get("THRESHOLD", "0.5"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
TRAIN_OUT = os.environ.get("TRAINING_OUTPUT_DIR", "/data/training_output")
SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_feats = None


def _auth(req: urllib.request.Request) -> urllib.request.Request:
    if TOKEN:
        req.add_header("X-Auth-Token", TOKEN)
    return req


def featurizer() -> AudioFeatures:
    global _feats
    if _feats is None:
        _feats = AudioFeatures()
    return _feats


def fetch_clips(label: str, eval_only: bool) -> np.ndarray:
    """Pull labelled clips from the puck as (N, CLIP_LEN) int16.

    eval_only keeps just the held-out slice (for negatives the model trained on
    the rest); real positives are never trained on, so we take them all.
    """
    url = f"{ORCH}/events/export?label={label}"
    try:
        data = urllib.request.urlopen(url, timeout=30).read()
    except Exception as e:
        print(f"  ({label}) fetch failed: {e}")
        return np.empty((0, CLIP_LEN), np.int16)
    clips = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".wav"):
                continue
            clip_id = m.name.rsplit("/", 1)[-1][: -len(".wav")]
            if eval_only and not is_eval_clip(clip_id):
                continue
            with wave.open(tar.extractfile(m)) as w:
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            a = a[:CLIP_LEN] if len(a) >= CLIP_LEN else np.pad(a, (0, CLIP_LEN - len(a)))
            clips.append(a)
    return np.stack(clips).astype(np.int16) if clips else np.empty((0, CLIP_LEN), np.int16)


def embed(clips: np.ndarray) -> np.ndarray:
    if len(clips) == 0:
        return np.empty((0, 16, 96), np.float32)
    return featurizer().embed_clips(clips, batch_size=64).astype(np.float32)


def synthetic_positive_feats() -> np.ndarray:
    """openWakeWord writes its own held-out positive test features under the
    training output dir; use them as a synthetic-positive baseline if present."""
    globs = os.environ.get(
        "EVAL_SYNTH_GLOBS", "*positive*test*.npy,*positive*val*.npy"
    ).split(",")
    for pat in globs:
        for path in glob.glob(os.path.join(TRAIN_OUT, "**", pat.strip()), recursive=True):
            try:
                arr = np.load(path)
            except Exception:
                continue
            if arr.ndim == 3 and arr.shape[1:] == (16, 96):
                print(f"  synthetic positives: {arr.shape} from {path}")
                return arr.astype(np.float32)
    print("  synthetic positives: none found (real clips still measured)")
    return np.empty((0, 16, 96), np.float32)


def load_current_model(path: str) -> bool:
    """Download the live model off the puck into `path`. False if unavailable."""
    try:
        req = _auth(urllib.request.Request(f"{ORCH}/models/{NAME}/file"))
        data = urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"  current model unavailable ({e}); reporting candidate only")
        return False
    with open(path, "wb") as f:
        f.write(data)
    return True


def scores(model_path: str, feats: np.ndarray) -> np.ndarray:
    if len(feats) == 0:
        return np.empty((0,), np.float32)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: feats})[0]
    return np.asarray(out).reshape(-1)


def metrics(neg: np.ndarray, pos: np.ndarray, t: float) -> dict:
    fa = int((neg >= t).sum()) if len(neg) else 0
    miss = int((pos < t).sum()) if len(pos) else 0
    return {
        "threshold": round(t, 3),
        "false_accepts": fa,
        "false_accept_rate": round(fa / len(neg), 4) if len(neg) else None,
        "misses": miss,
        "miss_rate": round(miss / len(pos), 4) if len(pos) else None,
    }


def verdict(cand: dict, cur: dict | None) -> str:
    if cur is None:
        return "baseline (no deployed model to compare against)"
    df = cand["false_accepts"] - cur["false_accepts"]
    dm = cand["misses"] - cur["misses"]
    if dm > 0:
        return (f"RECALL REGRESSION: {dm} more miss(es) on real wake words "
                f"(false accepts {df:+d}). Review before deploying.")
    if df < 0:
        return f"BETTER: {-df} fewer false accept(s), misses {dm:+d}."
    if df == 0 and dm == 0:
        return "NO CHANGE at this threshold."
    return f"MIXED: false accepts {df:+d}, misses {dm:+d}."


def main() -> None:
    if not ORCH or not NAME:
        print("evaluate: ORCHESTRATOR_URL and MODEL_NAME required; skipping.")
        return
    cand_path = os.path.join(OUTPUT_DIR, f"{NAME}.onnx")
    if not os.path.exists(cand_path):
        print(f"evaluate: candidate {cand_path} not found; skipping.")
        return

    print("=== Building held-out evaluation sets ===")
    neg = embed(fetch_clips("false", eval_only=True))
    pos_real = embed(fetch_clips("real", eval_only=False))
    pos_synth = synthetic_positive_feats()
    pos = np.concatenate([p for p in (pos_real, pos_synth) if len(p)]) \
        if (len(pos_real) or len(pos_synth)) else np.empty((0, 16, 96), np.float32)
    print(f"  held-out false wakes: {len(neg)}   real positives: {len(pos_real)}   "
          f"synthetic positives: {len(pos_synth)}")
    if len(neg) == 0 and len(pos) == 0:
        print("  no evaluation data available yet; skipping (label some clips first).")
        return

    cur_path = os.path.join(OUTPUT_DIR, f"{NAME}.current.onnx")
    have_current = load_current_model(cur_path)

    cand_neg, cand_pos = scores(cand_path, neg), scores(cand_path, pos)
    cur_neg, cur_pos = (scores(cur_path, neg), scores(cur_path, pos)) if have_current else (None, None)

    cand_m = metrics(cand_neg, cand_pos, THRESHOLD)
    cur_m = metrics(cur_neg, cur_pos, THRESHOLD) if have_current else None
    v = verdict(cand_m, cur_m)

    fa_n = len(neg)
    ms_n = len(pos)

    def row(label: str, m: dict) -> str:
        fa = f"{m['false_accepts']}/{fa_n}"
        ms = f"{m['misses']}/{ms_n}"
        return f"{label:<12}{fa:>16}{ms:>10}"

    print(f"\n=== Evaluation @ threshold {THRESHOLD} ===")
    print(f"{'model':<12}{'false_accepts':>16}{'misses':>10}")
    print(row("candidate", cand_m))
    if cur_m:
        print(row("current", cur_m))
    print(f"\nVERDICT: {v}")

    print("\n=== Candidate false-accept / miss tradeoff (threshold sweep) ===")
    print(f"{'thr':>5}{'false_accepts':>16}{'misses':>10}")
    sweep = []
    for t in SWEEP:
        mt = metrics(cand_neg, cand_pos, t)
        sweep.append(mt)
        fa = f"{mt['false_accepts']}/{fa_n}"
        ms = f"{mt['misses']}/{ms_n}"
        print(f"{t:>5}{fa:>16}{ms:>10}")

    report = {
        "model": NAME,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": THRESHOLD,
        "sets": {"held_out_false": len(neg), "real_positive": len(pos_real),
                 "synthetic_positive": int(len(pos_synth))},
        "candidate": cand_m,
        "current": cur_m,
        "verdict": v,
        "candidate_sweep": sweep,
    }
    report_path = os.path.join(OUTPUT_DIR, f"{NAME}.eval.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")
    if os.path.exists(cur_path):
        os.remove(cur_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Evaluation is advisory; never fail the training run over it.
        print(f"evaluate: error ({e}); continuing.", file=sys.stderr)
