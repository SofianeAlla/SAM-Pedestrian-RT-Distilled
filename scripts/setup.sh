#!/usr/bin/env bash
# One-time setup for the pedestrian expert prototype.
# Usage:
#   bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Installing Python deps ==="
pip install --user -e ".[sam3,trt]"

echo
echo "=== Verifying GPU + key imports ==="
python -c "
import torch
print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device: {torch.cuda.get_device_name(0)}')

from ultralytics import YOLO
import ultralytics
print(f'ultralytics {ultralytics.__version__}')

try:
    from transformers import Sam3Processor, Sam3Model
    print('transformers SAM 3 API: OK')
except ImportError as e:
    print(f'transformers SAM 3 API: MISSING ({e})')
    print('  Run: pip install --upgrade transformers')
"

echo
echo "=== Fetching demo video ==="
python scripts/fetch_sample_video.py --out-dir data/sample_videos

echo
echo "Setup complete. Next: huggingface-cli login   then   bash scripts/run_smoke.sh"
