"""Download a person-only subset of COCO val2017 for SAM 3 distillation.

Pulls images individually from `images.cocodataset.org` based on the
official annotations, so we only download what we need (instead of the
full 1 GB val2017 zip). Filters to images that contain at least
`min_persons` annotated person bboxes and have reasonable diversity
(skips near-duplicates by image_id stride). Writes to:

  out_dir/
    train/                ~500 images
    val/                  ~50 images
    image_ids.json        provenance

The COCO annotations are themselves not used for training — SAM 3
re-annotates with text prompts. COCO is just the *image source*. This
keeps the demo video (Intel sample-videos) cleanly held out.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"


def download_to_bytes(url: str, label: str) -> bytes:
    print(f"Downloading {label}: {url}")
    req = Request(url, headers={"User-Agent": "pedestrian-expert/0.1"})
    with urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunks: list[bytes] = []
        wrote = 0
        while True:
            buf = resp.read(1 << 17)
            if not buf:
                break
            chunks.append(buf)
            wrote += len(buf)
            if total:
                pct = 100.0 * wrote / total
                sys.stdout.write(
                    f"\r  {label}: {wrote/1e6:6.1f} / {total/1e6:6.1f} MB ({pct:5.1f}%)"
                )
                sys.stdout.flush()
        sys.stdout.write("\n")
    return b"".join(chunks)


def fetch_image(file_name: str, dest: Path, retries: int = 2) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = IMAGE_BASE_URL + file_name
    for attempt in range(retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "pedestrian-expert/0.1"})
            with urlopen(req, timeout=30) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt == retries:
                print(f"  fail {file_name}: {e}")
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/coco_person")
    p.add_argument("--train-n", type=int, default=500)
    p.add_argument("--val-n", type=int, default=50)
    p.add_argument("--min-persons", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--ann-cache",
        default="data/coco_person/_annotations_val2017.json",
        help="Reuse a previously downloaded annotations file",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ann_cache = Path(args.ann_cache)
    if ann_cache.exists():
        print(f"Using cached annotations: {ann_cache}")
        ann_data = json.loads(ann_cache.read_text())
    else:
        zip_bytes = download_to_bytes(
            ANNOTATIONS_URL, "annotations_trainval2017.zip"
        )
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            inner = "annotations/instances_val2017.json"
            with z.open(inner) as f:
                ann_data = json.load(f)
        ann_cache.parent.mkdir(parents=True, exist_ok=True)
        ann_cache.write_text(json.dumps(ann_data))

    # Find person category id (typically 1 in COCO).
    cats = {c["name"]: c["id"] for c in ann_data["categories"]}
    person_cat_id = cats.get("person")
    if person_cat_id is None:
        print("No 'person' category in COCO annotations.")
        return 1
    print(f"COCO person category id: {person_cat_id}")

    # Count person annotations per image.
    person_counts: dict[int, int] = {}
    for a in ann_data["annotations"]:
        if a["category_id"] != person_cat_id:
            continue
        person_counts[a["image_id"]] = person_counts.get(a["image_id"], 0) + 1

    eligible = [
        img
        for img in ann_data["images"]
        if person_counts.get(img["id"], 0) >= args.min_persons
    ]
    # Sort by descending person count to favor pedestrian-rich scenes,
    # but stride-sample for diversity (don't take only the densest).
    eligible.sort(key=lambda im: -person_counts[im["id"]])
    n_total = args.train_n + args.val_n
    if len(eligible) < n_total:
        print(
            f"Only {len(eligible)} eligible images "
            f"(needed {n_total}); using all available."
        )
        n_total = len(eligible)
        # rebalance.
        args.train_n = int(n_total * args.train_n / (args.train_n + args.val_n))
        args.val_n = n_total - args.train_n

    stride = max(1, len(eligible) // n_total)
    selected = eligible[::stride][:n_total]
    train_set = selected[: args.train_n]
    val_set = selected[args.train_n :]

    print(
        f"Selected {len(train_set)} train + {len(val_set)} val out of "
        f"{len(eligible)} person-containing val2017 images."
    )

    # Parallel download.
    def _job(img_meta: dict, dest: Path) -> tuple[str, bool]:
        file_name = img_meta["file_name"]
        ok = fetch_image(file_name, dest / file_name)
        return file_name, ok

    n_ok = 0
    n_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for im in train_set:
            futures.append(ex.submit(_job, im, train_dir))
        for im in val_set:
            futures.append(ex.submit(_job, im, val_dir))
        for i, fut in enumerate(as_completed(futures), start=1):
            name, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            if i % 50 == 0 or i == len(futures):
                print(f"  downloaded {i}/{len(futures)}  ok={n_ok} fail={n_fail}")

    # Provenance.
    (out_dir / "image_ids.json").write_text(
        json.dumps(
            {
                "train": [im["id"] for im in train_set],
                "val": [im["id"] for im in val_set],
                "person_cat_id": person_cat_id,
                "min_persons": args.min_persons,
            },
            indent=2,
        )
    )

    print(f"\nDone. {n_ok} ok / {n_fail} fail.")
    print(f"  train -> {train_dir}")
    print(f"  val   -> {val_dir}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
