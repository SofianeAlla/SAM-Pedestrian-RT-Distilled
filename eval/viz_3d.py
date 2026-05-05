"""Stage F — render a 3D viz of one keyframe.

Produces a single-frame 3-panel image and a 10-15 s rotating-camera
flythrough video showing:

  Panel 1 (top-left)   : Front camera with SAM 3 pedestrian masks overlaid.
  Panel 2 (top-right)  : Lidar points colored by pseudo-label
                         (gray = non-ped, orange = pedestrian),
                         nuScenes GT 3D pedestrian boxes drawn in green.
  Panel 3 (bottom)     : Rotating-camera lidar flythrough video as MP4.

This is the conversation-starter artifact: one screen tells the whole
story end-to-end (camera oracle -> lifted lidar labels -> ground truth
overlay).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

PED_CATEGORIES = (
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
    "human.pedestrian.stroller",
    "human.pedestrian.wheelchair",
    "human.pedestrian.personal_mobility",
)


def overlay_sam3_masks(image_bgr: np.ndarray, masks: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay binary instance masks (n_inst, H, W) onto an image."""
    out = image_bgr.copy()
    if masks.size == 0:
        return out
    union = np.zeros(image_bgr.shape[:2], dtype=bool)
    for i in range(masks.shape[0]):
        union |= masks[i] > 0
    if not union.any():
        return out
    color_layer = np.zeros_like(out)
    color_layer[union] = (50, 220, 255)  # warm yellow-orange in BGR
    return cv2.addWeighted(out, 1.0, color_layer, alpha, 0)


def render_lidar_bev(
    points: np.ndarray,
    pseudo_label: np.ndarray,
    gt_boxes_lidar: list,
    extent_m: float = 50.0,
    px_per_m: int = 8,
) -> np.ndarray:
    """Top-down BEV image. Points are (N, 4) in lidar frame.

    Pseudo-pedestrian points are drawn orange, non-ped points light gray,
    and GT pedestrian boxes drawn in green outline.
    """
    side_px = int(extent_m * 2 * px_per_m)
    img = np.full((side_px, side_px, 3), 30, dtype=np.uint8)

    # Project x (forward) up the image, y (left) to the right.
    cx = side_px // 2
    cy = side_px // 2

    def to_pixel(xy: np.ndarray) -> np.ndarray:
        u = (cx - xy[:, 1] * px_per_m).astype(np.int32)
        v = (cy - xy[:, 0] * px_per_m).astype(np.int32)
        return np.stack([u, v], axis=1)

    in_extent = (
        (np.abs(points[:, 0]) <= extent_m)
        & (np.abs(points[:, 1]) <= extent_m)
    )
    p = points[in_extent]
    lab = pseudo_label[in_extent]
    px = to_pixel(p[:, :2])

    valid = (px[:, 0] >= 0) & (px[:, 0] < side_px) & (px[:, 1] >= 0) & (px[:, 1] < side_px)
    px = px[valid]
    lab = lab[valid]

    non_ped = lab == 0
    ped = lab == 1
    img[px[non_ped, 1], px[non_ped, 0]] = (200, 200, 200)
    img[px[ped, 1], px[ped, 0]] = (50, 220, 255)

    # Ego marker.
    cv2.circle(img, (cx, cy), 5, (0, 200, 0), -1)
    cv2.line(img, (cx, cy), (cx, cy - 18), (0, 200, 0), 2)

    # GT boxes - draw their footprint.
    for box in gt_boxes_lidar:
        corners = box.corners()  # (3, 8)
        xy = corners[:2, :].T
        if (np.abs(xy) > extent_m).any():
            continue
        pix = to_pixel(xy)
        # bottom face indices in nuScenes Box: 2,3,7,6
        bottom = pix[[2, 3, 7, 6, 2]]
        cv2.polylines(img, [bottom.reshape(-1, 1, 2)], False, (0, 255, 0), 2)

    # Scale legend.
    cv2.putText(
        img, f"+/-{int(extent_m)} m",
        (10, side_px - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA,
    )
    return img


def make_panel(
    front_with_masks: np.ndarray,
    bev: np.ndarray,
    headline: str,
) -> np.ndarray:
    """3-panel layout:
       top:    front camera with SAM 3 masks (resized to bev width)
       bottom: BEV
       header: 30 px black bar with headline text.
    """
    H_HEADER = 36
    bev_h, bev_w = bev.shape[:2]
    cam_h, cam_w = front_with_masks.shape[:2]
    target_w = bev_w
    target_cam_h = int(cam_h * target_w / cam_w)
    cam_resized = cv2.resize(front_with_masks, (target_w, target_cam_h))
    out = np.zeros(
        (H_HEADER + target_cam_h + bev_h, target_w, 3), dtype=np.uint8
    )
    cv2.putText(
        out, headline,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    out[H_HEADER : H_HEADER + target_cam_h] = cam_resized
    out[H_HEADER + target_cam_h :] = bev
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--pseudo-dir", default="data/nuscenes_pseudo_3d")
    p.add_argument("--masks-dir", default="data/nuscenes_sam3_masks")
    p.add_argument("--sample-token", default=None,
                   help="Specific keyframe token; default = first w/ peds")
    p.add_argument("--out-image", default="outputs/demo_3d_panel.png")
    p.add_argument("--out-video", default="outputs/demo_3d.mp4")
    p.add_argument("--n-frames", type=int, default=120)
    args = p.parse_args()

    print("Loading nuscenes-devkit...")
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion

    nusc = NuScenes(
        version=args.version, dataroot=args.dataroot, verbose=False
    )

    pseudo_dir = Path(args.pseudo_dir)
    masks_dir = Path(args.masks_dir)

    # Pick a sample.
    if args.sample_token:
        sample = nusc.get("sample", args.sample_token)
    else:
        chosen = None
        for s in nusc.sample:
            cache = pseudo_dir / f"{s['token']}.npz"
            if not cache.exists():
                continue
            with np.load(cache) as d:
                if int(d["label"].sum()) > 0:
                    chosen = s
                    break
        if chosen is None:
            print("No sample with positive pseudo-pedestrian points.")
            return 1
        sample = chosen
    print(f"Sample token: {sample['token']}")

    # Load lidar + pseudo-label.
    cache = pseudo_dir / f"{sample['token']}.npz"
    with np.load(cache, allow_pickle=True) as d:
        points = d["points"]  # (N, 4) lidar frame
        pseudo_label = d["label"]
        confidence = d["confidence"]

    # GT pedestrian boxes in lidar frame.
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_record = nusc.get("sample_data", lidar_token)
    cs = nusc.get(
        "calibrated_sensor", lidar_record["calibrated_sensor_token"]
    )
    ego = nusc.get("ego_pose", lidar_record["ego_pose_token"])
    R_e_w_inv = Quaternion(ego["rotation"]).inverse
    R_sl_e_inv = Quaternion(cs["rotation"]).inverse

    gt_boxes_lidar = []
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
        box.rotate(R_e_w_inv)
        box.translate(-np.array(cs["translation"]))
        box.rotate(R_sl_e_inv)
        gt_boxes_lidar.append(box)

    # Front camera + SAM 3 mask overlay.
    cam_token = sample["data"]["CAM_FRONT"]
    cam_record = nusc.get("sample_data", cam_token)
    img_path = str(Path(args.dataroot) / cam_record["filename"])
    front = cv2.imread(img_path)
    masks_cache = masks_dir / sample["token"] / "CAM_FRONT.npz"
    if masks_cache.exists():
        with np.load(masks_cache, allow_pickle=True) as d:
            front_masks = d["masks"]
    else:
        front_masks = np.zeros((0, *front.shape[:2]), dtype=np.uint8)
    front_with_masks = overlay_sam3_masks(front, front_masks)

    bev = render_lidar_bev(points, pseudo_label, gt_boxes_lidar)
    headline = (
        f"sample {sample['token'][:8]}  "
        f"lidar pts={points.shape[0]}  "
        f"pseudo-ped pts={int(pseudo_label.sum())}  "
        f"GT boxes={len(gt_boxes_lidar)}"
    )
    panel = make_panel(front_with_masks, bev, headline)

    Path(args.out_image).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out_image, panel)
    print(f"Wrote {args.out_image}  shape={panel.shape}")

    # Rotating BEV flythrough.
    Path(args.out_video).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fh, fw = panel.shape[:2]
    writer = cv2.VideoWriter(args.out_video, fourcc, 24, (fw, fh))
    for t in range(args.n_frames):
        # Rotate BEV by spinning the points before drawing.
        theta = 2 * np.pi * t / args.n_frames
        R = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta),  np.cos(theta), 0],
             [0, 0, 1]],
            dtype=np.float32,
        )
        rotated_pts = points.copy()
        rotated_pts[:, :3] = points[:, :3] @ R.T
        rotated_boxes = []
        for box in gt_boxes_lidar:
            from nuscenes.utils.data_classes import Box as _B  # local import
            from copy import deepcopy
            b = deepcopy(box)
            b.rotate(Quaternion(axis=[0, 0, 1], angle=theta))
            rotated_boxes.append(b)
        bev_t = render_lidar_bev(rotated_pts, pseudo_label, rotated_boxes)
        panel_t = make_panel(front_with_masks, bev_t, headline)
        writer.write(panel_t)
    writer.release()
    print(f"Wrote {args.out_video}  ({args.n_frames} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
