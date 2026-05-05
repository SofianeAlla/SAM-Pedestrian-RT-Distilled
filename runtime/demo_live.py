"""Run the pedestrian expert over a video and write an annotated MP4.

This is the demo deliverable. Takes a video file (or webcam index),
runs the pedestrian expert per frame, draws boxes + masks + a latency
overlay, and encodes the output to MP4.

Example:
    python -m runtime.demo_live \
        --weights yolov8n-seg.pt \
        --video data/sample_drive.mp4 \
        --output outputs/demo_output.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from runtime.pedestrian_expert import InferenceResult, PedestrianExpert

PEDESTRIAN_COLOR = (50, 220, 255)  # BGR — warm yellow-orange
MASK_ALPHA = 0.45
TEXT_COLOR = (255, 255, 255)


def draw_result(
    frame: np.ndarray,
    result: InferenceResult,
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """Draw boxes + masks + a HUD on a copy of the frame."""
    out = frame.copy()
    h, w = frame.shape[:2]

    if result.detections:
        # Stack masks into a per-pixel boolean union for the alpha overlay.
        union = np.zeros((h, w), dtype=bool)
        for det in result.detections:
            if det.mask is None:
                continue
            m = det.mask
            if m.shape != (h, w):
                m = cv2.resize(
                    m, (w, h), interpolation=cv2.INTER_LINEAR
                )
            union |= m > 0.5

        if union.any():
            color_layer = np.zeros_like(out)
            color_layer[union] = PEDESTRIAN_COLOR
            out = cv2.addWeighted(out, 1.0, color_layer, MASK_ALPHA, 0)

        for det in result.detections:
            x1, y1, x2, y2 = (int(v) for v in det.box_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), PEDESTRIAN_COLOR, 2)
            label = f"{det.class_name} {det.score:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                out, (x1, y1 - th - 6), (x1 + tw + 4, y1), PEDESTRIAN_COLOR, -1
            )
            cv2.putText(
                out,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    hud = (
        f"Frame {frame_idx:6d}  "
        f"{result.latency_ms:5.1f} ms  "
        f"{fps:5.1f} FPS  "
        f"peds={len(result.detections)}"
    )
    cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(
        out, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA
    )
    cv2.putText(
        out,
        "SAM 3-distilled pedestrian expert",
        (8, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return out


def open_source(src: str) -> cv2.VideoCapture:
    """Open a video file or webcam index."""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")
    return cap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="YOLO .pt weights")
    p.add_argument("--video", required=True, help="Video file path or webcam index")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all")
    p.add_argument("--fp32", action="store_true", help="Disable FP16")
    args = p.parse_args()

    expert = PedestrianExpert(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=not args.fp32,
    )

    cap = open_source(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    frame_idx = 0
    ema_fps = 0.0
    t_loop_start = time.perf_counter()
    total_latency = 0.0
    total_dets = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            result = expert.infer(frame)
            inst_fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            ema_fps = inst_fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst_fps

            annotated = draw_result(frame, result, ema_fps, frame_idx)
            writer.write(annotated)

            total_latency += result.latency_ms
            total_dets += len(result.detections)
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_loop_start
    if frame_idx == 0:
        print("No frames processed.")
        return 1

    avg_lat = total_latency / frame_idx
    avg_fps = frame_idx / max(elapsed, 1e-6)
    print(f"Wrote {out_path}")
    print(f"Frames:           {frame_idx}")
    print(f"Avg inference:    {avg_lat:.2f} ms")
    print(f"Avg loop FPS:     {avg_fps:.2f}")
    print(f"Total detections: {total_dets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
