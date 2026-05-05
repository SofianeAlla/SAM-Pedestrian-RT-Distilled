"""Download + extract nuScenes mini (v1.0-mini, ~4 GB).

Public CloudFront mirror, no auth required.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz"


def download(url: str, dest: Path, chunk: int = 1 << 17) -> None:
    print(f"Fetching {url}")
    req = Request(url, headers={"User-Agent": "pedestrian-expert/0.1"})
    with urlopen(req, timeout=120) as resp:
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
                        f"\r  {wrote/1e9:5.2f} / {total/1e9:5.2f} GB ({pct:5.1f}%)"
                    )
                    sys.stdout.flush()
        sys.stdout.write("\n")


def extract(tgz: Path, dest: Path) -> None:
    print(f"Extracting {tgz} -> {dest}")
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        default="data/nuscenes",
        help="Where v1.0-mini/ will be created.",
    )
    p.add_argument(
        "--keep-tar",
        action="store_true",
        help="Keep the .tgz after extraction.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tgz = out_dir / "v1.0-mini.tgz"

    if not tgz.exists() or tgz.stat().st_size < 4_000_000_000:
        download(URL, tgz)

    extract(tgz, out_dir)
    if not args.keep_tar:
        tgz.unlink(missing_ok=True)
    print(f"Done: {out_dir}/v1.0-mini")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
