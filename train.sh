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

# ── Optional: pull the puck's flagged false wakes as hard negatives ──────────
if [ -n "${ORCHESTRATOR_URL:-}" ] && [ -n "${PULL_NEGATIVES:-}" ]; then
    echo "=== Pulling false-positive hard negatives from $ORCHESTRATOR_URL ==="
    MODEL_NAME="$MODEL_NAME" python /app/pull_negatives.py
    echo ""
fi

# ── Step 5: Train model ─────────────────────────────────────────────────────
# ONNX only: the puck runs the .onnx. (--convert_to_tflite died on every run —
# the image's TF 2.8.1 can't run against its protobuf — and the `|| true`-style
# guard it needed also swallowed genuine training failures.) Under set -e a
# failed train.py now fails the run.
OUTPUT_DIR="/data/training_output"
# Remove the previous build first so a run can never "succeed" by re-copying a
# stale model left over from an earlier training.
rm -f "$OUTPUT_DIR/$MODEL_NAME.onnx"
echo "=== Step 5/5: Training model ==="
python openwakeword/train.py --training_config /app/config.yaml --train_model
echo ""

# ── Copy output model ────────────────────────────────────────────────────────
echo "=== Copying output model to /output/ ==="
mkdir -p /output
if [ ! -f "$OUTPUT_DIR/$MODEL_NAME.onnx" ]; then
    echo "  ERROR: train.py finished but $OUTPUT_DIR/$MODEL_NAME.onnx was not produced"
    exit 1
fi
cp "$OUTPUT_DIR/$MODEL_NAME.onnx" /output/
echo "  Copied $MODEL_NAME.onnx"
echo ""

# ── Evaluate the candidate against the currently-deployed model (advisory) ───
# Held-out false wakes + confirmed-real wakes the model never trained on, so the
# numbers reflect generalization. Report-only: never blocks the deploy below.
if [ -n "${ORCHESTRATOR_URL:-}" ]; then
    echo "=== Evaluating candidate vs. deployed model ==="
    MODEL_NAME="$MODEL_NAME" OUTPUT_DIR="/output" TRAINING_OUTPUT_DIR="$OUTPUT_DIR" \
        python /app/evaluate.py || echo "  (evaluation skipped/failed; continuing)"
    echo ""
fi

# ── Optional: deploy the new model to the orchestrator (hot-reloads on the puck) ─
if [ -n "${ORCHESTRATOR_URL:-}" ] && [ -n "${PUSH_MODEL:-}" ]; then
    ONNX="/output/${MODEL_NAME}.onnx"
    if [ -f "$ONNX" ]; then
        echo "=== Deploying $MODEL_NAME.onnx to $ORCHESTRATOR_URL ==="
        curl -sf -X POST "${ORCHESTRATOR_URL}/models/${MODEL_NAME}" \
            ${MODEL_PUSH_TOKEN:+-H "X-Auth-Token: ${MODEL_PUSH_TOKEN}"} \
            --data-binary "@${ONNX}" -w "\n" \
            && echo "  Deployed and hot-reloaded on the puck." \
            || echo "  Deploy failed (model still saved in /output/)."
    else
        echo "  Skipping deploy: $ONNX not found."
    fi
    echo ""
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "============================================"
echo "Output files:"
ls -lh /output/
