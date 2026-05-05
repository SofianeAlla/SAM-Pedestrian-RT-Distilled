"""Stage B — lift SAM 3 camera-space pedestrian masks to lidar points.

For every nuScenes keyframe:
  1. Load the lidar sweep, transform to ego frame at the lidar timestamp.
  2. For each of the 6 cameras:
     a. Compensate ego motion between camera and lidar timestamps.
     b. Project lidar points into the camera image plane.
     c. For points landing inside a SAM 3 pedestrian mask, accumulate
        a soft confidence
            confidence_C = score * geometry_factor(point, camera)
        where geometry_factor decays with depth and tilts to zero at
        the image edges.
  3. Multi-camera consensus: per-point confidence = max across cameras,
     binary label = (max_conf >= tau).

Outputs (per sample_token):
  data/nuscenes_pseudo_3d/{sample_token}.npz
    points        : float32 (N, 4)  -- x, y, z, intensity (lidar frame)
    label         : uint8 (N,)      -- 0 non-ped, 1 pedestrian
    confidence    : float32 (N,)    -- max across cameras, in [0, 1]
    cam_hits      : uint8 (N, 6)    -- which cameras saw each point
    n_in_mask     : uint8 (N,)      -- in how many cameras the point hit a mask

Lift-only metric (Stage E) consumes these files; the 3D student
(Stage C) consumes these as supervision.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def geometry_factor(
    depth: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    image_w: int,
    image_h: int,
    near: float = 1.0,
    far: float = 60.0,
    edge_pad: float = 0.05,
) -> np.ndarray:
    """Per-point confidence weight from projection geometry.

    1.0 at depths in [near, far/2], decays to ~0.3 at far. 1.0 in the
    central image area, fades to 0 within edge_pad fraction of border.
    """
    # Depth term.
    z = np.clip(depth, near, far)
    depth_w = np.where(
        z <= far / 2,
        1.0,
        1.0 - 0.7 * (z - far / 2) / (far - far / 2),
    )

    # Edge term.
    pad_x = edge_pad * image_w
    pad_y = edge_pad * image_h
    dx = np.minimum(u, image_w - u) / max(pad_x, 1.0)
    dy = np.minimum(v, image_h - v) / max(pad_y, 1.0)
    edge_w = np.clip(np.minimum(dx, dy), 0.0, 1.0)

    return depth_w * edge_w


def lift_one_sample(
    nusc,
    sample,
    masks_dir: Path,
    tau: float,
    near: float,
    far: float,
) -> dict | None:
    """Compute per-point pseudo-labels for one keyframe."""
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    sample_token = sample["token"]
    masks_sample_dir = masks_dir / sample_token
    if not masks_sample_dir.exists():
        return None

    # Lidar in lidar-sensor frame.
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_record = nusc.get("sample_data", lidar_token)
    pcl_path = os.path.join(nusc.dataroot, lidar_record["filename"])
    pc = LidarPointCloud.from_file(pcl_path)  # (4, N)
    points_sensor = pc.points.copy()
    n = points_sensor.shape[1]

    # sensor -> ego (at lidar t) -> world
    cs_lidar = nusc.get(
        "calibrated_sensor", lidar_record["calibrated_sensor_token"]
    )
    ego_lidar = nusc.get("ego_pose", lidar_record["ego_pose_token"])
    R_sl_e = Quaternion(cs_lidar["rotation"]).rotation_matrix
    t_sl_e = np.array(cs_lidar["translation"]).reshape(3, 1)
    R_e_w = Quaternion(ego_lidar["rotation"]).rotation_matrix
    t_e_w = np.array(ego_lidar["translation"]).reshape(3, 1)

    # World coords once.
    points_ego = R_sl_e @ points_sensor[:3] + t_sl_e
    points_world = R_e_w @ points_ego + t_e_w

    confidence = np.zeros(n, dtype=np.float32)
    cam_hits = np.zeros((n, len(CAMERAS)), dtype=np.uint8)
    n_in_mask = np.zeros(n, dtype=np.uint8)

    for ci, ch in enumerate(CAMERAS):
        cache_path = masks_sample_dir / f"{ch}.npz"
        if not cache_path.exists():
            continue
        with np.load(cache_path, allow_pickle=True) as data:
            masks = data["masks"]
            scores = data["scores"]
            image_w = int(data["image_w"])
            image_h = int(data["image_h"])
        if masks.shape[0] == 0:
            continue

        cam_token = sample["data"][ch]
        cam_record = nusc.get("sample_data", cam_token)
        cs_cam = nusc.get(
            "calibrated_sensor", cam_record["calibrated_sensor_token"]
        )
        ego_cam = nusc.get("ego_pose", cam_record["ego_pose_token"])

        # world -> ego(at cam t) -> sensor
        R_w_e = Quaternion(ego_cam["rotation"]).rotation_matrix.T
        t_w_e = -R_w_e @ np.array(ego_cam["translation"]).reshape(3, 1)
        R_e_sc = Quaternion(cs_cam["rotation"]).rotation_matrix.T
        t_e_sc = -R_e_sc @ np.array(cs_cam["translation"]).reshape(3, 1)

        pts_ego_at_cam = R_w_e @ points_world + t_w_e
        pts_cam = R_e_sc @ pts_ego_at_cam + t_e_sc

        depth = pts_cam[2, :]
        in_front = depth > 0.1
        if not in_front.any():
            continue

        K = np.asarray(cs_cam["camera_intrinsic"])
        uvw = view_points(pts_cam, K, normalize=True)
        u = uvw[0, :]
        v = uvw[1, :]

        in_image = (
            in_front
            & (u >= 0)
            & (u < image_w)
            & (v >= 0)
            & (v < image_h)
        )
        if not in_image.any():
            continue

        idx = np.nonzero(in_image)[0]
        u_i = u[idx].astype(np.int32)
        v_i = v[idx].astype(np.int32)
        d_i = depth[idx]

        # Combined mask across instances, but keep per-point max score.
        # masks is (n_inst, H, W) uint8. We want, per point, the max
        # SAM 3 score over instances whose mask contains the point.
        ped_score = np.zeros(len(idx), dtype=np.float32)
        for mi in range(masks.shape[0]):
            inside = masks[mi, v_i, u_i] > 0
            if not inside.any():
                continue
            cand = scores[mi]
            ped_score = np.maximum(ped_score, np.where(inside, cand, 0.0))

        if not (ped_score > 0).any():
            continue

        gw = geometry_factor(d_i, u_i, v_i, image_w, image_h, near=near, far=far)
        cam_conf = ped_score * gw

        confidence[idx] = np.maximum(confidence[idx], cam_conf)
        hit_mask = ped_score > 0
        cam_hits[idx[hit_mask], ci] = 1
        n_in_mask[idx[hit_mask]] = n_in_mask[idx[hit_mask]] + 1

    label = (confidence >= tau).astype(np.uint8)
    return {
        "points": points_sensor.T.astype(np.float32),  # (N, 4) in sensor frame
        "label": label,
        "confidence": confidence,
        "cam_hits": cam_hits,
        "n_in_mask": n_in_mask,
        "sample_token": sample_token,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--masks-dir", default="data/nuscenes_sam3_masks")
    p.add_argument("--out-dir", default="data/nuscenes_pseudo_3d")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--near", type=float, default=1.0)
    p.add_argument("--far", type=float, default=60.0)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()

    print("Loading nuscenes-devkit...")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False
    )

    masks_dir = Path(args.masks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = nusc.sample
    if args.max_samples:
        samples = samples[: args.max_samples]

    n_done = 0
    n_skipped = 0
    n_pos_total = 0
    n_pts_total = 0

    for i, sample in enumerate(samples):
        out_path = out_dir / f"{sample['token']}.npz"
        if out_path.exists():
            n_skipped += 1
            continue
        result = lift_one_sample(
            nusc,
            sample,
            masks_dir,
            tau=args.tau,
            near=args.near,
            far=args.far,
        )
        if result is None:
            print(f"  [{i+1}/{len(samples)}] {sample['token'][:8]}: no masks")
            continue

        np.savez_compressed(
            out_path,
            points=result["points"],
            label=result["label"],
            confidence=result["confidence"],
            cam_hits=result["cam_hits"],
            n_in_mask=result["n_in_mask"],
            sample_token=result["sample_token"],
        )
        n_pos = int(result["label"].sum())
        n_pts = int(result["label"].shape[0])
        n_pos_total += n_pos
        n_pts_total += n_pts
        n_done += 1
        if (i + 1) % 25 == 0 or (i + 1) == len(samples):
            print(
                f"  [{i+1}/{len(samples)}] {sample['token'][:8]}: "
                f"{n_pos}/{n_pts} pts ped"
            )

    summary = {
        "version": args.version,
        "tau": args.tau,
        "near": args.near,
        "far": args.far,
        "n_samples_done": n_done,
        "n_samples_skipped": n_skipped,
        "total_lidar_points": n_pts_total,
        "total_ped_points": n_pos_total,
        "ped_fraction": (n_pos_total / max(1, n_pts_total)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone.\n{json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
