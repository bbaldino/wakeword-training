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
    """Download one AudioSet tar, extract, and convert to 16kHz WAV."""
    output_dir = os.path.join(DATA_DIR, "audioset_16k")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 10:
        print(f"  AudioSet already present ({len(os.listdir(output_dir))} files), skipping")
        return

    import datasets

    tar_dir = os.path.join(DATA_DIR, "audioset_raw")
    os.makedirs(tar_dir, exist_ok=True)

    fname = "bal_train09.tar"
    tar_path = os.path.join(tar_dir, fname)

    if not os.path.exists(tar_path):
        print("  Downloading AudioSet tar (~2GB)...")
        link = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{fname}"
        subprocess.run(["wget", "-q", "--show-progress", "-O", tar_path, link], check=True)

    print("  Extracting AudioSet...")
    subprocess.run(["tar", "-xf", tar_path, "-C", tar_dir], check=True)

    print("  Converting AudioSet to 16kHz WAV...")
    os.makedirs(output_dir, exist_ok=True)
    audio_files = list(Path(tar_dir).glob("**/audio/**/*.flac"))
    if not audio_files:
        audio_files = list(Path(tar_dir).glob("**/*.flac"))

    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(f) for f in audio_files]})
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    for row in tqdm(audioset_dataset, desc="  AudioSet 16kHz"):
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
    print(f"  Saved {len(os.listdir(output_dir))} AudioSet files")


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
        "rudraml/fma", name="small", split="train", streaming=True
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
