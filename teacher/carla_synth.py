"""Capture pedestrian frames + ground-truth masks from a running CARLA sim.

CARLA was running on this machine when the project was set up. CARLA can
spawn pedestrians under arbitrary lighting/weather and emits perfect
ground-truth bounding boxes and instance masks via its semantic-camera
sensor. This is a free, unlimited supplement to SAM 3 pseudo-labels —
especially valuable for hard cases (night, rain, occlusion) where
public datasets are thin.

Outputs match the YOLO-Seg dataset layout used by sam3_autolabel.py so
the trainer can consume both sources interchangeably.

Note: this is a stub. Wiring it up requires the CARLA Python client
(`pip install carla==<version>`) matching the running CARLA server
version, plus a few minutes of actor-spawning code. Left as an
overnight task once the SAM 3 path is proven.

Example (once wired up):
    python -m teacher.carla_synth \
        --host 127.0.0.1 --port 2000 \
        --n-frames 1000 --weather night \
        --out data/carla_synth
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--n-frames", type=int, default=1000)
    p.add_argument("--weather", default="day")
    p.add_argument("--out", default="data/carla_synth")
    p.add_argument("--imgsz", type=int, default=720)
    args = p.parse_args()

    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    try:
        import carla  # type: ignore
    except ImportError:
        print(
            "CARLA Python client not installed.\n"
            "  Install version matching your running CARLA server, e.g.:\n"
            "    pip install carla==0.9.15\n"
            "Then re-run this script.\n"
            f"Args parsed: host={args.host} port={args.port} "
            f"n_frames={args.n_frames} weather={args.weather} out={out_dir}"
        )
        return 2

    print(f"Connecting to CARLA at {args.host}:{args.port} ...")
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected. Map: {world.get_map().name}")

    # Implementation outline (left as TODO for the overnight run):
    # 1. Configure weather preset by args.weather
    # 2. Spawn an RGB camera + a semantic-segmentation camera at the same
    #    transform (both child of an autopilot vehicle).
    # 3. Spawn N pedestrians along sidewalks; set walker AI controllers.
    # 4. Tick the world; per tick, capture both camera frames.
    # 5. Convert semantic-seg pedestrian-class pixels to instance masks
    #    using CARLA's instance-seg sensor (or by per-actor bounding-box
    #    projection if instance-seg unavailable on this CARLA version).
    # 6. Emit YOLO-Seg polygon labels for each pedestrian instance.
    # 7. Save image to out_dir/images, label to out_dir/labels.
    print("CARLA capture loop not yet implemented. See file docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
