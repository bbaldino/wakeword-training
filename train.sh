#!/bin/bash
set -euo pipefail

# ── Validate inputs ──────────────────────────────────────────────────────────
if [ -z "${WAKE_WORD:-}" ]; then
    echo "ERROR: WAKE_WORD environment variable is required"
    echo "Usage: docker run --gpus all -e WAKE_WORD=\"hey nanoclaw\" ..."
    exit 1
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
N_SAMPLES="${N_SAMPLES:-10000}"
N_SAMPLES_VAL="${N_SAMPLES_VAL:-2000}"
TRAINING_STEPS="${TRAINING_STEPS:-50000}"
LAYER_SIZE="${LAYER_SIZE:-32}"

MODEL_NAME=$(echo "$WAKE_WORD" | tr ' ' '_')

echo "============================================"
echo "  Wake Word Training"
echo "============================================"
echo "  Wake word:       $WAKE_WORD"
echo "  Model name:      $MODEL_NAME"
echo "  Train samples:   $N_SAMPLES"
echo "  Val samples:     $N_SAMPLES_VAL"
echo "  Training steps:  $TRAINING_STEPS"
echo "  Layer size:      $LAYER_SIZE"
echo "============================================"
echo ""

# ── Step 1: Download datasets ────────────────────────────────────────────────
echo "=== Step 1/5: Downloading datasets ==="
python /app/download_data.py
echo ""

# ── Step 2: Generate training config from template ───────────────────────────
echo "=== Step 2/5: Generating training config ==="
python -c "
import yaml

with open('/app/config.template.yml') as f:
    config = yaml.safe_load(f)

config['model_name'] = '${MODEL_NAME}'
config['target_phrase'] = ['${WAKE_WORD}']
config['n_samples'] = ${N_SAMPLES}
config['n_samples_val'] = ${N_SAMPLES_VAL}
config['steps'] = ${TRAINING_STEPS}
config['layer_size'] = ${LAYER_SIZE}

with open('/app/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Config written to /app/config.yaml')
"
echo ""

# ── Step 3: Generate synthetic clips ─────────────────────────────────────────
echo "=== Step 3/5: Generating synthetic clips ==="
cd /app/openwakeword
python openwakeword/train.py --training_config /app/config.yaml --generate_clips
echo ""

# ── Step 4: Augment clips with noise/reverb ──────────────────────────────────
echo "=== Step 4/5: Augmenting clips ==="
python openwakeword/train.py --training_config /app/config.yaml --augment_clips
echo ""

# ── Step 5: Train model ─────────────────────────────────────────────────────
echo "=== Step 5/5: Training model ==="
python openwakeword/train.py --training_config /app/config.yaml --train_model --convert_to_tflite
echo ""

# ── Copy output models ───────────────────────────────────────────────────────
echo "=== Copying output models to /output/ ==="
OUTPUT_DIR="/data/training_output"
mkdir -p /output

if [ -f "$OUTPUT_DIR/$MODEL_NAME.onnx" ]; then
    cp "$OUTPUT_DIR/$MODEL_NAME.onnx" /output/
    echo "  Copied $MODEL_NAME.onnx"
else
    echo "  WARNING: $MODEL_NAME.onnx not found"
fi

if [ -f "$OUTPUT_DIR/$MODEL_NAME.tflite" ]; then
    cp "$OUTPUT_DIR/$MODEL_NAME.tflite" /output/
    echo "  Copied $MODEL_NAME.tflite"
else
    echo "  WARNING: $MODEL_NAME.tflite not found (ONNX model is still usable)"
    echo "  TFLite conversion can fail due to tensorflow version constraints."
    echo "  You can convert manually later if needed."
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "============================================"
echo "Output files:"
ls -lh /output/
