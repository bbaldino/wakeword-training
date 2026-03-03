# Wake Word Training (Docker)

Train custom wake word models for [openwakeword](https://github.com/dscripka/openwakeword) using synthetic speech generation and GPU-accelerated training. Designed to run on an Unraid server with an NVIDIA GPU.

## Quick Start

```bash
# Build the image
docker build -t wakeword-trainer .

# Train a wake word
docker run --gpus all \
  -e WAKE_WORD="hey nanoclaw" \
  -v /mnt/user/ai/wakeword-data:/data \
  -v /mnt/user/ai/wakeword-output:/output \
  wakeword-trainer
```

After training, copy the `.tflite` (or `.onnx`) model from the output directory and load it in `voice_pipeline.py`:

```python
oww = Model(
    wakeword_models=["/path/to/hey_nanoclaw.tflite"],
    inference_framework="tflite"
)
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WAKE_WORD` | **(required)** | Target phrase, e.g. `"hey nanoclaw"` |
| `N_SAMPLES` | `10000` | Number of positive training samples to generate |
| `N_SAMPLES_VAL` | `2000` | Number of validation samples |
| `TRAINING_STEPS` | `50000` | Maximum training steps |
| `LAYER_SIZE` | `32` | Model layer dimension (larger = more capable but slower) |

## Volume Mounts

| Mount Point | Purpose |
|-------------|---------|
| `/data` | Cached datasets and intermediate training data (~12GB first run) |
| `/output` | Where the trained `.onnx` and `.tflite` models are written |

The `/data` volume persists across runs so datasets only download once. Intermediate generated clips are also cached per wake word, so re-running with the same wake word skips clip generation.

## What It Does

The training pipeline has 5 stages:

1. **Download datasets** — room impulse responses (MIT), background noise (AudioSet), music (FMA), and pre-computed negative features (ACAV100M ~8GB, validation ~1.5GB)
2. **Generate config** — merges environment variables into `config.template.yml`
3. **Generate synthetic clips** — uses Piper TTS to create thousands of clips of the wake word (and adversarial near-misses) across many synthetic voices
4. **Augment clips** — applies room reverb, background noise, and other augmentations to make synthetic clips more realistic
5. **Train model** — trains a small DNN classifier and exports to ONNX + TFLite

## Hardware Requirements

- NVIDIA GPU with CUDA 12.1 support (tested on RTX 3090)
- ~15GB disk for `/data` on first run (cached after that)
- Training itself uses modest GPU memory; the 3090's 24GB is more than sufficient

## Customizing the Config

For advanced tuning, edit `config.template.yml` before building. Key fields:

- `max_negative_weight`: Controls false-positive reduction aggressiveness (default: 1500)
- `target_false_positives_per_hour`: Target FP rate (default: 0.2)
- `augmentation_rounds`: Reuse each synthetic clip N times with different augmentation (default: 1)
- `batch_n_per_class`: Per-class batch sizes during training

See `openwakeword/examples/custom_model.yml` for full documentation of all fields.

## Troubleshooting

**TFLite conversion fails**: The `.onnx` model is still produced and works fine with openwakeword using `inference_framework="onnx"`. TFLite conversion depends on `tensorflow-cpu==2.8.1` which has strict dependency requirements.

**Out of memory during clip generation**: Reduce `tts_batch_size` in `config.template.yml`.

**Training seems stuck**: The auto-training process uses cyclic learning rates and adaptive batching. Long plateaus are normal. Check GPU utilization with `nvidia-smi`.

## Future Enhancements

- Support for augmenting training data with real voice recordings (mount at `/voice-samples`)
- Custom verifier model for speaker-specific wake word detection
