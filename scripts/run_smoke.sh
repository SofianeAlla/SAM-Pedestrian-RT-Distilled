#!/usr/bin/env bash
# Smoke-test the pipeline end-to-end on a tiny seed corpus.
# Designed for the user's RTX 4070 Laptop (8 GB).
# Usage:
#   bash scripts/run_smoke.sh
#
# Prereq: HF SAM 3 access granted + `huggingface-cli login` done.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED_DIR="data/seed_images"
PSEUDO_DIR="data/pseudo_labels"
RUN_NAME="ped_smoke"

mkdir -p "$SEED_DIR" "$PSEUDO_DIR" outputs

if [ -z "$(ls -A "$SEED_DIR" 2>/dev/null)" ]; then
  echo "Seed image dir empty: $SEED_DIR"
  echo "Drop ~50-200 driving images there (Cityscapes val, BDD subset, or webcam grabs)"
  echo "and re-run."
  exit 1
fi

echo "=== Stage 1: SAM 3 pseudo-labels ==="
python -m teacher.sam3_autolabel \
  --images "$SEED_DIR" \
  --out "$PSEUDO_DIR" \
  --threshold 0.5 \
  --save-masks

echo
echo "=== Stage 2: filter cyclist/scooter overlaps ==="
python -m teacher.pseudolabel_filter --dataset "$PSEUDO_DIR" --iou-threshold 0.4

echo
echo "=== Stage 3: brief distillation fine-tune ==="
python -m distill.train --config distill/configs/ped_yolov8n.yaml --run-name "$RUN_NAME"

BEST="runs/$RUN_NAME/weights/best.pt"
if [ ! -f "$BEST" ]; then
  echo "Training did not produce $BEST"
  exit 2
fi

echo
echo "=== Stage 4: demo video ==="
PYTHONPATH="$PWD" python -m runtime.demo_live \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --output outputs/demo_smoke.mp4 \
  --imgsz 640 --conf 0.3

echo
echo "=== Stage 5: benchmark ==="
PYTHONPATH="$PWD" python -m eval.benchmark \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --max-frames 300

echo
echo "Done. Smoke-run artifacts:"
echo "  weights -> $BEST"
echo "  demo    -> outputs/demo_smoke.mp4"
