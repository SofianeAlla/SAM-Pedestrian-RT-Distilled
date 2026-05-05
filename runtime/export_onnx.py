"""Export a trained YOLO checkpoint to ONNX for downstream TensorRT build.

Writes the ONNX file alongside the .pt by default. Uses Ultralytics'
built-in exporter which handles dynamic axes and seg-head op fusion.

Example:
    python -m runtime.export_onnx --weights runs/train/exp/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="YOLO .pt weights")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--half", action="store_true", help="Export FP16")
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--dynamic", action="store_true", help="Dynamic batch dim")
    args = p.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(weights)

    model = YOLO(str(weights))
    out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        half=args.half,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )
    print(f"Exported ONNX -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
