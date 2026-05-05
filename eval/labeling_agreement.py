"""Stage E — labeling agreement vs nuScenes 3D pedestrian box GT.

Tests the *lift itself* (Stage B output) against nuScenes' human 3D
labels, without ever training a downstream detector.

For each keyframe:
  1. Pull the SAM 3 / lift output (data/nuscenes_pseudo_3d/{token}.npz).
  2. Pull nuScenes 3D pedestrian boxes for the same keyframe (filtered
     to category 'human.pedestrian.*').
  3. Convert each box to its enclosed lidar points (in lidar frame).
  4. Build per-point GT label: 1 if the point is inside any pedestrian
     box, else 0.
  5. Compare against pseudo-label: report point-level
        precision, recall, F1
        per-distance breakdown (0-15 m, 15-30 m, 30+ m)
        confidence-vs-correctness curve

This metric is meaningful at nuScenes-mini scale where headline 3D
detection mAP is too noisy to trust.

Outputs:
  outputs/labeling_agreement.json    — headline numbers
  outputs/labeling_agreement.png     — distance breakdown bar chart
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PED_CATEGORIES = (
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "human.pedestrian.personal_mobility",
)


def points_in_box(box, points: np.ndarray) -> np.ndarray:
    """Return boolean mask of which points lie inside `box`.

    `box` is a nuscenes Box in the same frame as `points` ((3, N))."""
    from nuscenes.utils.geometry_utils import points_in_box as _pib

    return _pib(box, points[:3, :])


def boxes_for_sample(nusc, sample) -> list:
    """All 3D pedestrian boxes for a sample, in lidar frame."""
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_record = nusc.get("sample_data", lidar_token)
    cs = nusc.get(
        "calibrated_sensor", lidar_record["calibrated_sensor_token"]
    )
    ego = nusc.get("ego_pose", lidar_record["ego_pose_token"])

    R_e_w = Quaternion(ego["rotation"]).rotation_matrix
    t_e_w = np.array(ego["translation"])
    R_sl_e = Quaternion(cs["rotation"]).rotation_matrix
    t_sl_e = np.array(cs["translation"])

    boxes_lidar = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["category_name"] not in PED_CATEGORIES:
            continue
        # Box is in world frame originally; transform world -> ego -> sensor.
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
            name=ann["category_name"],
        )
        # world -> ego
        box.translate(-t_e_w)
        box.rotate(Quaternion(ego["rotation"]).inverse)
        # ego -> sensor
        box.translate(-t_sl_e)
        box.rotate(Quaternion(cs["rotation"]).inverse)
        boxes_lidar.append(box)
    return boxes_lidar


def per_point_gt(boxes, points: np.ndarray) -> np.ndarray:
    """OR of point-in-any-pedestrian-box, returns uint8 (N,)."""
    n = points.shape[1]
    inside = np.zeros(n, dtype=bool)
    for b in boxes:
        inside |= points_in_box(b, points)
    return inside.astype(np.uint8)


def evaluate_agreement(
    pseudo_label: np.ndarray,
    pseudo_conf: np.ndarray,
    gt_label: np.ndarray,
    distance: np.ndarray,
    distance_bins: tuple[tuple[float, float], ...] = (
        (0.0, 15.0),
        (15.0, 30.0),
        (30.0, float("inf")),
    ),
) -> dict:
    """Per-point P/R/F1 overall and per-distance bin."""
    metrics: dict = {}

    def stats(p: np.ndarray, g: np.ndarray) -> dict:
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        return {
            "n": int(p.shape[0]),
            "n_gt_pos": int(g.sum()),
            "n_pseudo_pos": int(p.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    metrics["overall"] = stats(pseudo_label, gt_label)

    bin_metrics = {}
    for lo, hi in distance_bins:
        mask = (distance >= lo) & (distance < hi)
        if not mask.any():
            continue
        bin_metrics[f"{int(lo)}-{'inf' if hi == float('inf') else int(hi)}m"] = stats(
            pseudo_label[mask], gt_label[mask]
        )
    metrics["by_distance"] = bin_metrics
    return metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--pseudo-dir", default="data/nuscenes_pseudo_3d")
    p.add_argument("--out", default="outputs/labeling_agreement.json")
    p.add_argument("--plot", default="outputs/labeling_agreement.png")
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    print("Loading nuscenes-devkit...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False
    )
    samples = nusc.sample
    if args.max_samples:
        samples = samples[: args.max_samples]

    pseudo_dir = Path(args.pseudo_dir)

    # Aggregate over all samples by concatenation.
    all_pseudo: list[np.ndarray] = []
    all_pseudo_conf: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    all_dist: list[np.ndarray] = []
    n_done = 0
    n_missing = 0

    for i, sample in enumerate(samples):
        cache = pseudo_dir / f"{sample['token']}.npz"
        if not cache.exists():
            n_missing += 1
            continue
        with np.load(cache, allow_pickle=True) as data:
            points = data["points"]  # (N, 4) in sensor frame
            pseudo_label = data["label"]
            pseudo_conf = data["confidence"]

        # Build GT per-point labels.
        boxes = boxes_for_sample(nusc, sample)
        gt = per_point_gt(boxes, points.T)

        dist = np.linalg.norm(points[:, :2], axis=1)

        all_pseudo.append(pseudo_label)
        all_pseudo_conf.append(pseudo_conf)
        all_gt.append(gt)
        all_dist.append(dist)
        n_done += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1}/{len(samples)}] aggregated {n_done} samples")

    if not all_pseudo:
        print("No pseudo-label files found.")
        return 1

    pseudo = np.concatenate(all_pseudo)
    pseudo_conf = np.concatenate(all_pseudo_conf)
    gt = np.concatenate(all_gt)
    dist = np.concatenate(all_dist)

    metrics = evaluate_agreement(pseudo, pseudo_conf, gt, dist)
    metrics["n_samples_evaluated"] = n_done
    metrics["n_samples_missing_pseudo"] = n_missing

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n{json.dumps(metrics, indent=2)}")
    print(f"\nWrote {out_path}")

    # Bar chart of P/R/F1 per distance bin.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = list(metrics["by_distance"].keys())
        prec = [metrics["by_distance"][b]["precision"] for b in bins]
        rec = [metrics["by_distance"][b]["recall"] for b in bins]
        f1 = [metrics["by_distance"][b]["f1"] for b in bins]
        x = np.arange(len(bins))
        width = 0.27
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - width, prec, width, label="precision")
        ax.bar(x, rec, width, label="recall")
        ax.bar(x + width, f1, width, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(
            "SAM 3 -> lidar pseudo-label vs nuScenes GT (point-level)\n"
            f"overall F1={metrics['overall']['f1']:.3f}, "
            f"P={metrics['overall']['precision']:.3f}, "
            f"R={metrics['overall']['recall']:.3f}, "
            f"n_samples={n_done}"
        )
        ax.legend()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Wrote {args.plot}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
