#!/usr/bin/env bash
# Full overnight training run.
# Same pipeline as run_smoke.sh but with a larger corpus and full epochs.
# Usage:
#   bash scripts/run_full.sh
#
# Expects the user to have:
#   - Populated data/seed_images/ with ~10K-20K real driving images
#     (Cityscapes train + EuroCity Persons + NightOwls + CrowdHuman + BDD subset)
#   - Run teacher/carla_synth.py to add synthetic frames at data/carla_synth/
#   - HF SAM 3 access via `huggingface-cli login`
set -euo pipefail
cd "$(dirname "$0")/.."

SEED_DIR="data/seed_images"
SYNTH_DIR="data/carla_synth/images"
PSEUDO_DIR="data/pseudo_labels"
RUN_NAME="ped_full"

mkdir -p "$PSEUDO_DIR" outputs

n_real=$(ls "$SEED_DIR" 2>/dev/null | wc -l)
echo "Real images:      $n_real"
if [ -d "$SYNTH_DIR" ]; then
  n_synth=$(ls "$SYNTH_DIR" 2>/dev/null | wc -l)
  echo "Synthetic frames: $n_synth"
fi

echo
echo "=== Stage 1: SAM 3 pseudo-labels (full) ==="
python -m teacher.sam3_autolabel \
  --images "$SEED_DIR" \
  --out "$PSEUDO_DIR" \
  --threshold 0.5 \
  --save-masks

echo
echo "=== Stage 2: filter cyclist/scooter overlaps ==="
python -m teacher.pseudolabel_filter --dataset "$PSEUDO_DIR"

echo
echo "=== Stage 3: full distillation training ==="
# Override smoke epochs/batch with full-run values.
python -m distill.train \
  --config distill/configs/ped_yolov8n.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --run-name "$RUN_NAME"

BEST="runs/$RUN_NAME/weights/best.pt"

echo
echo "=== Stage 4: demo video ==="
PYTHONPATH="$PWD" python -m runtime.demo_live \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --output outputs/demo_full.mp4 \
  --imgsz 640 --conf 0.3

echo
echo "=== Stage 5: ONNX export + desktop TensorRT engine ==="
PYTHONPATH="$PWD" python -m runtime.export_onnx --weights "$BEST" --imgsz 640
ONNX="${BEST%.pt}.onnx"
if [ -f "$ONNX" ]; then
  PYTHONPATH="$PWD" python -m runtime.build_trt \
    --onnx "$ONNX" \
    --output "${BEST%.pt}.engine" --fp16 || true
fi

echo
echo "=== Stage 6: benchmark ==="
PYTHONPATH="$PWD" python -m eval.benchmark \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --max-frames 1000

echo
echo "Full-run artifacts:"
echo "  weights -> $BEST"
echo "  demo    -> outputs/demo_full.mp4"
