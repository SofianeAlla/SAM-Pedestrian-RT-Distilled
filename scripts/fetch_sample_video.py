"""Download a permissively-licensed sample pedestrian video for the demo.

Pulls from Intel's sample-videos repo (CC license, stable URLs).
Skips the download if the file already exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

CANDIDATES = [
    (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
        "person-bicycle-car-detection.mp4",
        "person-bicycle-car-detection.mp4",
    ),
    (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
        "people-detection.mp4",
        "people-detection.mp4",
    ),
    (
        "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
        "head-pose-face-detection-female-and-male.mp4",
        "head-pose-fallback.mp4",
    ),
]


def fetch(url: str, dest: Path, chunk: int = 1 << 16) -> bool:
    try:
        with urlopen(url, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            wrote = 0
            with open(dest, "wb") as f:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    wrote += len(buf)
                    if total:
                        pct = 100.0 * wrote / total
                        sys.stdout.write(
                            f"\r  {dest.name}: {wrote/1e6:5.1f} / {total/1e6:5.1f} MB ({pct:5.1f}%)"
                        )
                        sys.stdout.flush()
            sys.stdout.write("\n")
            return True
    except Exception as e:
        print(f"  failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="data/sample_videos",
        help="Where to save the video",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for url, name in CANDIDATES:
        dest = out_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"Already have {dest}")
            print(dest)
            return 0
        print(f"Trying {url}")
        if fetch(url, dest):
            print(f"OK -> {dest}")
            print(dest)
            return 0

    print("All candidates failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
