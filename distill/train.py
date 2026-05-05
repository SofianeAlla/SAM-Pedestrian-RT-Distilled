"""Distill the pedestrian student from SAM 3 pseudo-labels.

Wraps Ultralytics' YOLOv8-Seg trainer. The 'distillation' here is data-
distillation: the student (YOLOv8n-Seg) is trained on SAM 3 teacher's
pseudo-labels. Initialization from a pretrained COCO checkpoint
preserves the 'person' visual prior; SAM 3 supervision sharpens mask
quality, especially on long-tail cases (occlusion, night, crowd).

For an even tighter knowledge-transfer setup later, this script can be
extended with feature-map / logit KD (hooks into the YOLO backbone +
a frozen teacher forward), but the data-distill path alone is a
strong baseline and the standard SAM-to-YOLO distillation recipe.

Example:
    python -m distill.train --config distill/configs/ped_yolov8n.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="YAML training config")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--imgsz", type=int, default=None, help="Override imgsz")
    p.add_argument("--batch", type=int, default=None, help="Override batch")
    p.add_argument("--run-name", default=None, help="Override run name")
    args = p.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    from ultralytics import YOLO

    init_weights = cfg["init_weights"]
    print(f"Loading initial weights: {init_weights}")
    model = YOLO(init_weights)

    print(f"Training on dataset: {cfg['dataset_yaml']}")
    print(
        f"  epochs={cfg['epochs']} batch={cfg['batch']} "
        f"imgsz={cfg['imgsz']} device={cfg['device']}"
    )

    results = model.train(
        data=cfg["dataset_yaml"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        imgsz=cfg["imgsz"],
        device=cfg["device"],
        project=cfg["project"],
        name=cfg["run_name"],
        single_cls=cfg.get("single_cls", True),
        amp=cfg.get("amp", True),
        patience=cfg.get("patience", 5),
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.001),
        lrf=cfg.get("lrf", 0.01),
        warmup_epochs=cfg.get("warmup_epochs", 1.0),
        mosaic=cfg.get("mosaic", 0.5),
        close_mosaic=cfg.get("close_mosaic", 5),
        val=cfg.get("val", True),
        save_period=cfg.get("save_period", -1),
        workers=cfg.get("workers", 0),
        exist_ok=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir is not None:
        print(f"Run dir: {save_dir}")
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            print(f"Best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
