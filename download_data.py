#!/usr/bin/env python3
"""Download and prepare datasets required for openwakeword training.

All datasets are saved to /data/ and cached across container runs via volume mount.
"""

import os
import subprocess
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from tqdm import tqdm

DATA_DIR = os.environ.get("DATA_DIR", "/data")


def download_mit_rirs():
    """Download MIT room impulse responses from HuggingFace."""
    output_dir = os.path.join(DATA_DIR, "mit_rirs")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"  MIT RIRs already present ({len(os.listdir(output_dir))} files), skipping")
        return

    import datasets

    print("  Downloading MIT room impulse responses...")
    os.makedirs(output_dir, exist_ok=True)
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    count = 0
    for row in tqdm(rir_dataset, desc="  MIT RIRs"):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
        count += 1
    print(f"  Saved {count} RIR files")


def download_audioset():
    """Download AudioSet balanced-train split via HuggingFace streaming and convert to 16kHz WAV."""
    output_dir = os.path.join(DATA_DIR, "audioset_16k")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 10:
        print(f"  AudioSet already present ({len(os.listdir(output_dir))} files), skipping")
        return

    import datasets

    print("  Streaming AudioSet balanced-train split from HuggingFace...")
    os.makedirs(output_dir, exist_ok=True)

    audioset = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train", streaming=True
    )
    audioset = audioset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    count = 0
    for row in tqdm(audioset, desc="  AudioSet 16kHz"):
        audio = row["audio"]
        name = f"audioset_{count:06d}.wav"
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (audio["array"] * 32767).astype(np.int16),
        )
        count += 1
    print(f"  Saved {count} AudioSet files")


def download_fma():
    """Download 1 hour of Free Music Archive clips from HuggingFace."""
    output_dir = os.path.join(DATA_DIR, "fma")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 10:
        print(f"  FMA already present ({len(os.listdir(output_dir))} files), skipping")
        return

    import datasets

    print("  Downloading FMA music dataset (1 hour subset)...")
    os.makedirs(output_dir, exist_ok=True)

    fma_dataset = datasets.load_dataset(
        "rudraml/fma", name="small", split="train", streaming=True, trust_remote_code=True
    )
    fma_dataset = iter(
        fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    )

    n_hours = 1
    n_clips = n_hours * 3600 // 30  # FMA clips are 30 seconds each

    for i in tqdm(range(n_clips), desc="  FMA clips"):
        row = next(fma_dataset)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    print(f"  Saved {len(os.listdir(output_dir))} FMA clips")


def download_features():
    """Download pre-computed openwakeword features for training and validation."""
    files = {
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": (
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
            "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        ),
        "validation_set_features.npy": (
            "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/"
            "validation_set_features.npy"
        ),
    }

    for filename, url in files.items():
        dest = os.path.join(DATA_DIR, filename)
        if os.path.exists(dest):
            print(f"  {filename} already present, skipping")
            continue
        print(f"  Downloading {filename}...")
        subprocess.run(["wget", "-q", "--show-progress", "-O", dest, url], check=True)


if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n[1/4] Room impulse responses")
    download_mit_rirs()

    print("\n[2/4] AudioSet background noise")
    download_audioset()

    print("\n[3/4] FMA music")
    download_fma()

    print("\n[4/4] Pre-computed features (ACAV100M + validation)")
    download_features()

    print("\nAll datasets ready.")
