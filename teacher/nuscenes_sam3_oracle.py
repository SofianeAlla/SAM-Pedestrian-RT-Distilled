"""Stage A — run SAM 3 over the 6 cameras of every nuScenes keyframe.

Inputs: a nuScenes (mini or full) dataroot.
Outputs: per-camera mask cache under data/nuscenes_sam3_masks/.

For each (sample_token, channel) pair we save:
  data/nuscenes_sam3_masks/{sample_token}/{channel}.npz
    masks   : uint8 (n_inst, H, W)         — binary instance masks
    scores  : float32 (n_inst,)            — SAM 3 score per mask
    prompts : list[str]                    — which prompt produced each mask
    image_w : int
    image_h : int

The cyclist filter (teacher.pseudolabel_filter equivalent) is run inline
here so we don't need a second pass: any pedestrian mask whose bbox
overlaps a YOLOv8n bicycle/motorcycle detection above iou_threshold is
dropped before saving.

Compute budget: ~400 keyframes * 6 cams * ~1.5 s/img on a 4070 (FP16)
~ 60 minutes total. The cache is content-addressed by sample_token so
the script is resumable.

Example
-------
    HF_TOKEN=... python -m teacher.nuscenes_sam3_oracle \\
        --dataroot data/nuscenes/v1.0-mini \\
        --version v1.0-mini \\
        --out-dir data/nuscenes_sam3_masks
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from teacher.sam3_autolabel import (
    DEFAULT_POSITIVE_PROMPTS,
    load_sam3,
    run_sam3_on_image,
)

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def mask_to_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = (mask > 0).nonzero()
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def iou_xyxy(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aw = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bw = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1, aw + bw - inter)


def filter_cyclists(
    detections: list[tuple[np.ndarray, float, str]],
    image_bgr: np.ndarray,
    cyclist_detector: Any,
    iou_threshold: float,
    device: str,
) -> list[tuple[np.ndarray, float, str]]:
    """Drop pedestrian masks that overlap a 2-wheeler bbox."""
    res = cyclist_detector.predict(
        image_bgr, classes=[1, 3], device=device, verbose=False, conf=0.25
    )
    wheeler_boxes: list[tuple[float, ...]] = []
    if res and res[0].boxes is not None and len(res[0].boxes) > 0:
        for b in res[0].boxes.xyxy.cpu().numpy():
            wheeler_boxes.append(tuple(map(float, b)))

    kept: list[tuple[np.ndarray, float, str]] = []
    for mask, score, prompt in detections:
        bb = mask_to_xyxy(mask)
        if bb is None:
            continue
        is_wheeler = any(
            iou_xyxy(bb, wb) >= iou_threshold for wb in wheeler_boxes
        )
        if not is_wheeler:
            kept.append((mask, score, prompt))
    return kept


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataroot",
        required=True,
        help="nuScenes root (e.g. data/nuscenes/v1.0-mini)",
    )
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--out-dir", default="data/nuscenes_sam3_masks")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--cyclist-iou", type=float, default=0.4)
    p.add_argument(
        "--cyclist-detector",
        default="yolov8n.pt",
        help="Detector for 2-wheeler filter; uses Ultralytics weights.",
    )
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip (sample, channel) pairs that already have a cache file.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading nuscenes-devkit...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False
    )
    print(f"  {len(nusc.sample)} samples")

    print("Loading SAM 3...")
    sam3 = load_sam3()
    if sam3 is None:
        print("SAM 3 not loaded. Set HF_TOKEN and accept facebook/sam3 license.")
        return 1
    processor, model, device = sam3

    print("Loading cyclist detector...")
    from ultralytics import YOLO

    cyclist_det = YOLO(args.cyclist_detector)

    n_done = 0
    n_skipped = 0
    n_total_detections = 0

    samples = nusc.sample
    if args.max_samples:
        samples = samples[: args.max_samples]
    n_pairs = len(samples) * len(CAMERAS)
    pair_idx = 0

    for s_idx, sample in enumerate(samples):
        sample_token = sample["token"]
        sample_dir = out_dir / sample_token
        sample_dir.mkdir(parents=True, exist_ok=True)

        for ch in CAMERAS:
            pair_idx += 1
            cache_path = sample_dir / f"{ch}.npz"
            if args.skip_existing and cache_path.exists():
                n_skipped += 1
                continue

            cam_token = sample["data"][ch]
            cam_record = nusc.get("sample_data", cam_token)
            img_path = os.path.join(args.dataroot, cam_record["filename"])
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"  [{pair_idx}/{n_pairs}] {sample_token[:8]} {ch}: unreadable")
                continue
            h, w = image_bgr.shape[:2]

            dets = run_sam3_on_image(
                processor,
                model,
                device,
                image_bgr,
                DEFAULT_POSITIVE_PROMPTS,
                threshold=args.threshold,
            )
            dets = filter_cyclists(
                dets, image_bgr, cyclist_det, args.cyclist_iou, device="cuda:0"
            )

            if not dets:
                np.savez_compressed(
                    cache_path,
                    masks=np.zeros((0, h, w), dtype=np.uint8),
                    scores=np.zeros((0,), dtype=np.float32),
                    prompts=np.array([], dtype=object),
                    image_w=np.int32(w),
                    image_h=np.int32(h),
                )
            else:
                masks_arr = np.stack(
                    [m.astype(np.uint8) for m, _, _ in dets], axis=0
                )
                scores_arr = np.array([s for _, s, _ in dets], dtype=np.float32)
                prompts_arr = np.array([p for _, _, p in dets], dtype=object)
                np.savez_compressed(
                    cache_path,
                    masks=masks_arr,
                    scores=scores_arr,
                    prompts=prompts_arr,
                    image_w=np.int32(w),
                    image_h=np.int32(h),
                )
                n_total_detections += len(dets)

            n_done += 1
            if pair_idx % 10 == 0 or pair_idx == n_pairs:
                print(
                    f"  [{pair_idx}/{n_pairs}] sample {s_idx+1}/{len(samples)} "
                    f"{ch}: {len(dets)} dets  "
                    f"(done {n_done}, skipped {n_skipped})"
                )

    summary = {
        "version": args.version,
        "n_samples": len(samples),
        "n_pairs_total": n_pairs,
        "n_pairs_done": n_done,
        "n_pairs_skipped": n_skipped,
        "n_total_detections": n_total_detections,
        "threshold": args.threshold,
        "cyclist_iou": args.cyclist_iou,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
