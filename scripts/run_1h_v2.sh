#!/usr/bin/env bash
# 1-hour SAM 3 distillation run, with proper data hygiene.
#
#   TRAIN+VAL : COCO val2017 person subset (~500 + 50 diverse images).
#               Student has never seen these.
#   DEMO/EVAL : Intel sample-videos pedestrian clip. Held out, zero
#               overlap with training. Same goes for the benchmark.
#
# Architecture:
#   * dataset YAML declares all 80 COCO class names. yolov8n-seg.pt's
#     pretrained head is therefore preserved; only class 0 (person)
#     carries gradient from SAM 3 pseudo-labels.
#   * freeze=10 freezes the entire YOLOv8n backbone (the future-MoE
#     shared substrate). Only the neck + head are refined.
#   * single_cls=False so the 80-class head structure is retained.
#
# Wall time, RTX 4070 Laptop, 8 GB:
#   ~3 min  COCO subset download
#   ~16 min SAM 3 over ~550 images
#   ~3 min  cyclist filter
#   ~25 min train (50 epochs, batch 8)
#   ~2 min  demo + benchmark
#   ~ 49 min total
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"

echo '=== Stage 1: download COCO val2017 person subset ==='
python scripts/build_coco_person_subset.py \
  --out-dir data/coco_person --train-n 500 --val-n 50 --workers 12

echo
echo '=== Stage 2: SAM 3 pseudo-labels (train split) ==='
PYTHONPATH="$PWD" python -m teacher.sam3_autolabel \
  --images data/coco_person/train \
  --out data/pseudo_labels_train \
  --threshold 0.45

echo
echo '=== Stage 3: SAM 3 pseudo-labels (val split) ==='
PYTHONPATH="$PWD" python -m teacher.sam3_autolabel \
  --images data/coco_person/val \
  --out data/pseudo_labels_val \
  --threshold 0.45

echo
echo '=== Stage 4: filter cyclist/scooter overlaps ==='
PYTHONPATH="$PWD" python -m teacher.pseudolabel_filter \
  --dataset data/pseudo_labels_train --iou-threshold 0.4
PYTHONPATH="$PWD" python -m teacher.pseudolabel_filter \
  --dataset data/pseudo_labels_val --iou-threshold 0.4

echo
echo '=== Stage 5: build canonical YOLO dataset (80 COCO names) ==='
python scripts/build_distill_dataset.py \
  --train-src data/pseudo_labels_train \
  --val-src data/pseudo_labels_val \
  --out-dir data/coco_distill

echo
echo '=== Stage 6: distillation training (50 epochs, frozen backbone) ==='
python -c "
from ultralytics import YOLO
m = YOLO('yolov8n-seg.pt')
r = m.train(
    data='data/coco_distill/dataset.yaml',
    epochs=50,
    batch=8,
    imgsz=640,
    device=0,
    project='runs',
    name='ped_1h',
    single_cls=False,
    freeze=10,
    amp=True,
    patience=15,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=2.0,
    mosaic=0.5,
    close_mosaic=10,
    val=True,
    save_period=10,
    workers=0,
    exist_ok=True,
)
print('RUN_DIR=' + str(r.save_dir))
"

BEST=$(find runs -name best.pt 2>/dev/null | grep ped_1h | head -1)
if [ -z "$BEST" ]; then
  echo 'best.pt not found under runs/'
  exit 2
fi
echo "Best weights: $BEST"

echo
echo '=== Stage 7: demo on HELD-OUT Intel video ==='
PYTHONPATH="$PWD" python -m runtime.demo_live \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --output outputs/demo_1h.mp4 \
  --imgsz 640 --conf 0.25

echo
echo '=== Stage 8: benchmark on the same held-out video ==='
PYTHONPATH="$PWD" python -m eval.benchmark \
  --weights "$BEST" \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --max-frames 500

echo
echo 'DONE.'
echo "  weights -> $BEST"
echo '  demo    -> outputs/demo_1h.mp4'
