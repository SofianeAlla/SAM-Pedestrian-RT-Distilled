"""Stage G — diagnostic plot of where the lift fails.

Two plots:

  1. F1 vs camera-coverage: F1 score on points seen by k cameras
     (k = 0, 1, 2, 3+). Hypothesis: more cameras = higher F1.
  2. Confidence-vs-correctness curve: histogram of pseudo confidence
     for TP, FP, FN, TN. Tells us if a higher tau would improve
     precision at acceptable recall cost.

Both use the same per-point pseudo + GT pairing as
eval.labeling_agreement, but slice on different axes.
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


def per_point_gt(boxes, points: np.ndarray) -> np.ndarray:
    from nuscenes.utils.geometry_utils import points_in_box

    n = points.shape[1]
    inside = np.zeros(n, dtype=bool)
    for b in boxes:
        inside |= points_in_box(b, points[:3, :])
    return inside.astype(np.uint8)


def boxes_for_sample(nusc, sample) -> list:
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_record = nusc.get("sample_data", lidar_token)
    cs = nusc.get("calibrated_sensor", lidar_record["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", lidar_record["ego_pose_token"])

    boxes_lidar = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        if ann["category_name"] not in PED_CATEGORIES:
            continue
        box = Box(
            center=ann["translation"],
            size=ann["size"],
            orientation=Quaternion(ann["rotation"]),
            name=ann["category_name"],
        )
        box.translate(-np.array(ego["translation"]))
        box.rotate(Quaternion(ego["rotation"]).inverse)
        box.translate(-np.array(cs["translation"]))
        box.rotate(Quaternion(cs["rotation"]).inverse)
        boxes_lidar.append(box)
    return boxes_lidar


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--pseudo-dir", default="data/nuscenes_pseudo_3d")
    p.add_argument("--out-coverage", default="outputs/failure_mode_coverage.png")
    p.add_argument("--out-confidence", default="outputs/failure_mode_confidence.png")
    p.add_argument("--out-json", default="outputs/failure_mode.json")
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

    pseudo_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []
    cov_all: list[np.ndarray] = []

    n_done = 0
    for i, sample in enumerate(samples):
        cache = pseudo_dir / f"{sample['token']}.npz"
        if not cache.exists():
            continue
        with np.load(cache, allow_pickle=True) as data:
            points = data["points"]
            pseudo = data["label"]
            conf = data["confidence"]
            n_in_mask = data["n_in_mask"]
        boxes = boxes_for_sample(nusc, sample)
        gt = per_point_gt(boxes, points.T)
        pseudo_all.append(pseudo)
        conf_all.append(conf)
        gt_all.append(gt)
        cov_all.append(n_in_mask)
        n_done += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] aggregated {n_done}")

    if not pseudo_all:
        print("No pseudo-label files.")
        return 1

    pseudo = np.concatenate(pseudo_all)
    conf = np.concatenate(conf_all)
    gt = np.concatenate(gt_all)
    cov = np.concatenate(cov_all)

    # Coverage breakdown.
    coverage_buckets = {"0": cov == 0, "1": cov == 1, "2": cov == 2, "3+": cov >= 3}
    coverage_metrics = {}
    for k, m in coverage_buckets.items():
        if not m.any():
            continue
        p = pseudo[m]
        g = gt[m]
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        tn = int(((p == 0) & (g == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        coverage_metrics[k] = {
            "n": int(m.sum()),
            "n_gt_pos": int(g.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    # Confidence-vs-correctness.
    bins = np.linspace(0, 1, 21)
    conf_hists = {
        "tp": np.histogram(conf[(pseudo == 1) & (gt == 1)], bins=bins)[0].tolist(),
        "fp": np.histogram(conf[(pseudo == 1) & (gt == 0)], bins=bins)[0].tolist(),
        "fn_conf": np.histogram(conf[(pseudo == 0) & (gt == 1)], bins=bins)[0].tolist(),
    }

    out = {
        "n_samples": n_done,
        "by_camera_coverage": coverage_metrics,
        "confidence_histograms_bins": bins.tolist(),
        "confidence_histograms": conf_hists,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out_json}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Coverage plot.
        ks = list(coverage_metrics.keys())
        precs = [coverage_metrics[k]["precision"] for k in ks]
        recs = [coverage_metrics[k]["recall"] for k in ks]
        f1s = [coverage_metrics[k]["f1"] for k in ks]
        x = np.arange(len(ks))
        w = 0.27
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w, precs, w, label="precision")
        ax.bar(x, recs, w, label="recall")
        ax.bar(x + w, f1s, w, label="F1")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k} cam(s)" for k in ks])
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(
            "Lift quality vs how many cameras saw each lidar point\n"
            "(zero-camera bin = blind-spot points; ego-relative)"
        )
        ax.legend()
        Path(args.out_coverage).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_coverage, dpi=130)
        print(f"Wrote {args.out_coverage}")

        # Confidence histograms.
        centers = (bins[:-1] + bins[1:]) / 2
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(centers, conf_hists["tp"], width=0.04, alpha=0.7, label="TP",
               color="tab:green")
        ax.bar(centers, conf_hists["fp"], width=0.04, alpha=0.7, label="FP",
               color="tab:red")
        ax.set_xlim(0, 1)
        ax.set_xlabel("pseudo-label confidence")
        ax.set_ylabel("# points (log)")
        ax.set_yscale("log")
        ax.set_title(
            "Pseudo-confidence distribution: TP vs FP\n"
            "where the FP bar dominates is where a higher tau pays off"
        )
        ax.legend()
        Path(args.out_confidence).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.out_confidence, dpi=130)
        print(f"Wrote {args.out_confidence}")
    except Exception as e:
        print(f"Plot failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
