"""Benchmark the pedestrian expert on the local GPU.

Reports per-frame inference latency (mean / p50 / p95 / p99), throughput
in FPS, and rough power figure if `nvidia-smi` is available.

Example:
    python -m eval.benchmark --weights runs/ped_smoke/weights/best.pt \
        --video data/sample_videos/person-bicycle-car-detection.mp4
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np


def gpu_power_watts() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, ".")
    from runtime.pedestrian_expert import PedestrianExpert

    expert = PedestrianExpert(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=not args.fp32,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return 1

    # Warmup.
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        expert.infer(frame)

    latencies: list[float] = []
    start_pwr = gpu_power_watts()
    pwr_samples: list[float] = []
    if start_pwr is not None:
        pwr_samples.append(start_pwr)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = 0
    t0 = time.perf_counter()
    while n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        ti = time.perf_counter()
        expert.infer(frame)
        latencies.append((time.perf_counter() - ti) * 1000.0)
        n += 1
        if n % 50 == 0:
            p = gpu_power_watts()
            if p is not None:
                pwr_samples.append(p)
    elapsed = time.perf_counter() - t0
    cap.release()

    if not latencies:
        print("No frames benchmarked.")
        return 1

    latencies.sort()
    mean = statistics.mean(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    fps = n / elapsed

    print(f"Frames:         {n}")
    print(f"Mean latency:   {mean:.2f} ms")
    print(f"p50 latency:    {p50:.2f} ms")
    print(f"p95 latency:    {p95:.2f} ms")
    print(f"p99 latency:    {p99:.2f} ms")
    print(f"Throughput:     {fps:.2f} FPS")
    if pwr_samples:
        print(f"GPU power avg:  {statistics.mean(pwr_samples):.1f} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
