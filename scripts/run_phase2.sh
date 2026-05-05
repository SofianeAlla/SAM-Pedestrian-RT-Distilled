#!/usr/bin/env bash
# Phase 2 orchestrator: SAM 3 -> nuScenes camera masks -> 3D point pseudo-labels
# -> labeling agreement vs nuScenes GT -> 3D viz.
#
# Stage C (PointPillars training) is intentionally NOT run by this
# script - Windows install of OpenPCDet/mmdet3d is fragile and not
# guaranteed. Run train_3d.py manually if your env is set up.
#
# Wall time on RTX 4070 Laptop, 8 GB:
#   Stage A (SAM 3 over 6 cams * keyframes)   ~60 min
#   Stage B (lift)                            ~10 min
#   Stage E (labeling agreement)              ~2 min
#   Stage F (3D viz)                          ~1 min
#
# Requires HF_TOKEN with facebook/sam3 license accepted.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN before running this script}"

DATAROOT="data/nuscenes"   # contains samples/ sweeps/ maps/ v1.0-mini/ after extract
VERSION="v1.0-mini"

echo '=== Stage 0: ensure nuScenes mini available ==='
if [ ! -d "$DATAROOT/samples" ]; then
  python scripts/fetch_nuscenes_mini.py --out-dir "$DATAROOT"
fi

echo
echo '=== Stage A: SAM 3 over the 6 cameras ==='
PYTHONPATH="$PWD" python -m teacher.nuscenes_sam3_oracle \
  --dataroot "$DATAROOT" --version "$VERSION" \
  --out-dir data/nuscenes_sam3_masks

echo
echo '=== Stage B: 2D->3D lift ==='
PYTHONPATH="$PWD" python -m teacher.lidar_lift \
  --dataroot "$DATAROOT" --version "$VERSION" \
  --masks-dir data/nuscenes_sam3_masks \
  --out-dir data/nuscenes_pseudo_3d \
  --tau 0.5

echo
echo '=== Stage E: labeling agreement vs nuScenes GT ==='
PYTHONPATH="$PWD" python -m eval.labeling_agreement \
  --dataroot "$DATAROOT" --version "$VERSION" \
  --pseudo-dir data/nuscenes_pseudo_3d \
  --out outputs/labeling_agreement.json \
  --plot outputs/labeling_agreement.png

echo
echo '=== Stage F: 3D viz on a sample with pedestrians ==='
PYTHONPATH="$PWD" python -m eval.viz_3d \
  --dataroot "$DATAROOT" --version "$VERSION" \
  --pseudo-dir data/nuscenes_pseudo_3d \
  --masks-dir data/nuscenes_sam3_masks \
  --out-image outputs/demo_3d_panel.png \
  --out-video outputs/demo_3d.mp4

echo
echo 'DONE.'
echo '  outputs/labeling_agreement.json'
echo '  outputs/labeling_agreement.png'
echo '  outputs/demo_3d_panel.png'
echo '  outputs/demo_3d.mp4'
