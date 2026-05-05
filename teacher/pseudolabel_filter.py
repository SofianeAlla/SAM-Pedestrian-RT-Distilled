"""Post-hoc filter for SAM 3 pseudo-labels: keep person-on-foot only.

SAM 3's text prompting is good but imperfect — when given "pedestrian"
it occasionally returns cyclists or scooter riders too. This filter
does a heuristic clean-up:

1. Run a small bicycle/motorcycle/scooter detector over each image.
2. For each pedestrian box, compute IoU against any 2-wheeler box.
3. Drop the pedestrian if the overlap exceeds threshold (likely a rider).

This keeps the pedestrian expert's training set clean for the future
MoE separation between the pedestrian and cyclist experts.

Example:
    python -m teacher.pseudolabel_filter \
        --dataset data/pseudo_labels \
        --iou-threshold 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# COCO class indices for two-wheelers / scooter riders.
TWO_WHEELER_CLASSES = {
    1,  # bicycle
    3,  # motorcycle
}


def parse_yolo_seg_line(line: str) -> tuple[int, np.ndarray]:
    """Parse a YOLO-Seg label line into (class_id, polygon Nx2 normalised)."""
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    cid = int(parts[0])
    coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
    if coords.size % 2 != 0:
        raise ValueError("odd polygon coord count")
    return cid, coords.reshape(-1, 2)


def polygon_to_box_xyxy(poly_norm: np.ndarray, w: int, h: int) -> tuple[float, float, float, float]:
    pts = poly_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def iou_xyxy(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    b_area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="SAM 3 dataset dir")
    p.add_argument("--iou-threshold", type=float, default=0.4)
    p.add_argument("--detector", default="yolov8n.pt", help="2-wheeler detector")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    from ultralytics import YOLO

    ds = Path(args.dataset)
    images_dir = ds / "images"
    labels_dir = ds / "labels"
    if not labels_dir.exists():
        print(f"No labels dir at {labels_dir}")
        return 1

    print(f"Loading 2-wheeler detector: {args.detector}")
    det = YOLO(args.detector)

    n_files = 0
    n_kept_total = 0
    n_dropped_total = 0

    for label_path in sorted(labels_dir.glob("*.txt")):
        img_candidates = list(images_dir.glob(f"{label_path.stem}.*"))
        if not img_candidates:
            continue
        img = cv2.imread(str(img_candidates[0]))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Detect 2-wheelers in this image.
        results = det.predict(
            img, classes=list(TWO_WHEELER_CLASSES), device=args.device,
            verbose=False, conf=0.25,
        )
        wheeler_boxes: list[tuple[float, ...]] = []
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes.xyxy.cpu().numpy():
                    wheeler_boxes.append(tuple(map(float, b)))

        # Read pedestrian labels and filter.
        kept_lines: list[str] = []
        with open(label_path) as f:
            in_lines = f.readlines()
        for line in in_lines:
            try:
                _, poly = parse_yolo_seg_line(line)
            except ValueError:
                continue
            ped_box = polygon_to_box_xyxy(poly, w, h)
            drop = False
            for wb in wheeler_boxes:
                if iou_xyxy(ped_box, wb) >= args.iou_threshold:
                    drop = True
                    break
            if not drop:
                kept_lines.append(line)
                n_kept_total += 1
            else:
                n_dropped_total += 1

        # Overwrite the file.
        with open(label_path, "w") as f:
            f.writelines(kept_lines)
        n_files += 1

    print(
        f"Filtered {n_files} files. Kept {n_kept_total}, "
        f"dropped {n_dropped_total} ped-overlapping-wheeler labels."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
