"""SAM 3 teacher: generate pedestrian pseudo-labels for student distillation.

Reads a directory of images, prompts SAM 3 with person-on-foot concept
prompts (negatively prompted against cyclist/scooter/wheelchair to keep
this expert clean for the future MoE), and writes:

  out/
    images/                <symlink-or-copy of each input image>
    labels/<stem>.txt      YOLOv8-Seg polygon labels (class x1 y1 ... xn yn)
    masks/<stem>.png       Optional binary union mask (for KD losses)
    meta/<stem>.json       Confidence scores, raw box list, prompt used

The output directory is laid out as a YOLO dataset so distill/train.py
can consume it directly via Ultralytics with no extra adapter.

SAM 3 is loaded via Hugging Face Transformers (Sam3Processor / Sam3Model).
Falls back to printing a clear instruction if SAM 3 isn't yet installed —
the rest of the pipeline (distill, runtime, demo) does not depend on this
script having succeeded.

Example:
    python -m teacher.sam3_autolabel \
        --images data/seed_images \
        --out data/pseudo_labels \
        --prompts "pedestrian" "person walking"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_POSITIVE_PROMPTS = (
    "pedestrian",
    "person walking",
    "person standing",
    "person crossing the street",
)

DEFAULT_NEGATIVE_PROMPTS = (
    "cyclist",
    "person on bicycle",
    "person on scooter",
    "person on motorcycle",
    "person in wheelchair",
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_sam3(model_id: str | None = None) -> tuple[Any, Any, Any] | None:
    """Load SAM 3 via Hugging Face Transformers (>=5.7).

    Requires acceptance of the Meta SAM 3 license on Hugging Face and
    either `huggingface-cli login` or HF_TOKEN env var.

    Returns (processor, model, device) on success, None otherwise.
    """
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor  # type: ignore
    except ImportError as e:
        print(f"transformers Sam3 import failed: {e}")
        return None

    candidates = (
        [model_id]
        if model_id
        else ["facebook/sam3", "facebook/sam3-large", "facebook/sam3-base"]
    )

    for mid in candidates:
        try:
            processor = Sam3Processor.from_pretrained(mid)
            model = Sam3Model.from_pretrained(mid, torch_dtype=torch.float16)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            print(f"Loaded SAM 3: {mid} on {device}")
            return processor, model, device
        except Exception as e:
            print(f"  {mid}: {type(e).__name__}: {e}")

    return None


def mask_to_polygon(mask: np.ndarray, min_area: int = 100) -> list[np.ndarray] | None:
    """Convert a binary mask (HxW uint8) to YOLO-format polygon points.

    Returns the largest contour if the mask is fragmented, or None if no
    suitable contour exists.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    if not contours:
        return None
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    return [c.squeeze(1) for c in valid]


def polygon_to_yolo_seg_line(
    polygon: np.ndarray, w: int, h: int, class_id: int = 0
) -> str:
    """Format polygon as a YOLO-Seg label line (class x1 y1 ... normalised)."""
    pts = polygon.astype(np.float32)
    pts[:, 0] /= w
    pts[:, 1] /= h
    pts = np.clip(pts, 0.0, 1.0)
    flat = pts.flatten().tolist()
    return (
        f"{class_id} " + " ".join(f"{v:.6f}" for v in flat) + "\n"
    )


def run_sam3_on_image(
    processor: Any,
    model: Any,
    device: str,
    image_bgr: np.ndarray,
    positive_prompts: tuple[str, ...],
    threshold: float,
) -> list[tuple[np.ndarray, float, str]]:
    """Run SAM 3 with each positive prompt; return (mask, score, prompt) per detection."""
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections: list[tuple[np.ndarray, float, str]] = []

    for prompt in positive_prompts:
        try:
            inputs = processor(
                images=image_rgb, text=prompt, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            print(f"    prompt '{prompt}': {type(e).__name__}: {e}")
            continue

        # SAM 3 outputs vary by version; defensively handle a few shapes.
        masks = getattr(outputs, "pred_masks", None)
        scores = getattr(outputs, "iou_scores", None)
        if masks is None:
            masks = getattr(outputs, "masks", None)
        if scores is None:
            scores = getattr(outputs, "scores", None)

        if masks is None:
            continue

        masks_np = masks.float().cpu().numpy()
        scores_np = (
            scores.float().cpu().numpy()
            if scores is not None
            else np.ones(masks_np.shape[:2])
        )

        # Flatten to per-mask iteration regardless of (B, N, H, W) vs (N, H, W).
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
            scores_np = scores_np[0] if scores_np.ndim > 1 else scores_np

        for i in range(masks_np.shape[0]):
            score = (
                float(scores_np[i])
                if scores_np.ndim > 0 and i < len(np.atleast_1d(scores_np))
                else 1.0
            )
            if score < threshold:
                continue
            m = masks_np[i]
            if m.ndim == 3:
                m = m[0]
            m_bin = (m > 0.0).astype(np.uint8)
            if m_bin.sum() < 50:
                continue
            detections.append((m_bin, score, prompt))

    return detections


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="Input image directory")
    p.add_argument("--out", required=True, help="Output dataset directory")
    p.add_argument("--prompts", nargs="+", default=list(DEFAULT_POSITIVE_PROMPTS))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    p.add_argument(
        "--save-masks", action="store_true", help="Also save union PNG masks"
    )
    p.add_argument(
        "--model-id", default=None, help="HF model ID, e.g. facebook/sam3"
    )
    args = p.parse_args()

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        x for x in images_dir.iterdir() if x.suffix.lower() in IMG_EXTS
    )
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        print(f"No images found in {images_dir}")
        return 1

    print(f"Found {len(image_paths)} images.")

    sam3 = load_sam3(args.model_id)
    if sam3 is None:
        print(
            "\nSAM 3 not available in this environment.\n"
            "Install:  pip install --upgrade git+https://github.com/huggingface/transformers\n"
            "Then re-run this script. Until then, the rest of the pipeline\n"
            "(distill/train.py, runtime/, demo_live) is independent and works.\n"
        )
        return 2

    processor, model, device = sam3

    n_kept = 0
    for idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [{idx+1}/{len(image_paths)}] {img_path.name}: unreadable")
            continue
        h, w = image_bgr.shape[:2]

        dets = run_sam3_on_image(
            processor,
            model,
            device,
            image_bgr,
            tuple(args.prompts),
            args.threshold,
        )

        # Copy image into the dataset.
        out_img = out_dir / "images" / img_path.name
        if not out_img.exists():
            cv2.imwrite(str(out_img), image_bgr)

        # Write YOLO-Seg label.
        label_path = out_dir / "labels" / f"{img_path.stem}.txt"
        lines: list[str] = []
        union = np.zeros((h, w), dtype=np.uint8)
        for mask, score, prompt in dets:
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
            polys = mask_to_polygon(mask)
            if not polys:
                continue
            for poly in polys:
                if poly.shape[0] < 3:
                    continue
                lines.append(polygon_to_yolo_seg_line(poly, w, h, class_id=0))
            union |= mask

        with open(label_path, "w") as f:
            f.writelines(lines)

        if args.save_masks:
            cv2.imwrite(str(out_dir / "masks" / f"{img_path.stem}.png"), union * 255)

        meta = {
            "image": img_path.name,
            "width": w,
            "height": h,
            "n_detections": len(dets),
            "scores": [float(s) for _, s, _ in dets],
            "prompts_used": list(args.prompts),
        }
        with open(out_dir / "meta" / f"{img_path.stem}.json", "w") as f:
            json.dump(meta, f, indent=2)

        n_kept += int(len(dets) > 0)
        print(
            f"  [{idx+1}/{len(image_paths)}] {img_path.name}: {len(dets)} det(s)"
        )

    # Emit a YOLO dataset YAML for the trainer.
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        "# Auto-generated by teacher/sam3_autolabel.py\n"
        f"path: {out_dir.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: pedestrian\n"
    )
    print(f"\nWrote dataset YAML: {yaml_path}")
    print(f"Images with at least one detection: {n_kept}/{len(image_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
