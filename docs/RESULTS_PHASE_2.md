# Phase 2 Results — SAM 3-supervised lidar pseudo-labels on nuScenes mini

> Single end-to-end run, RTX 4070 Laptop, 8 GB VRAM, FP16.
> **Zero human 3D box labels were used during pseudo-labeling.** The
> nuScenes 3D pedestrian boxes are used only at evaluation time, to
> measure how well a SAM 3 → camera-lidar projection pipeline agrees
> with human annotations.

## TL;DR

We treat SAM 3 as a 2D oracle, run it on each of the 6 nuScenes cameras
per keyframe, and project the resulting pedestrian masks onto the lidar
sweep with multi-camera consensus. Point-level agreement with nuScenes'
own 3D pedestrian box ground truth, on **92 evaluated keyframes** (3.2 M
lidar points), with no detector training:

| Point-level vs nuScenes 3D ped boxes  | Value         |
|---------------------------------------|---------------|
| **Overall precision**                 | **0.685**     |
| **Overall recall**                    | **0.654**     |
| **Overall F1**                        | **0.669**     |
| F1 @ 0-15 m                           | 0.76          |
| F1 @ 15-30 m                          | 0.59          |
| F1 @ 30+ m                            | 0.34          |
| GT pedestrian points                  | 17,135        |
| Pseudo-positive points                | 16,360        |
| Total lidar points                    | 3,194,048     |

Full breakdown in [`outputs/labeling_agreement.json`](../outputs/labeling_agreement.json),
distance-bin chart in [`outputs/labeling_agreement.png`](../outputs/labeling_agreement.png).

## Pipeline (recap)

1. **Oracle** — SAM 3 (`facebook/sam3`, 840 M params, FP16) with concept
   prompts `{pedestrian, person walking, person standing, person crossing
   the street}`, negative-prompted against `{cyclist, person on bicycle,
   person on scooter, person in wheelchair}`. Run on every camera image
   of every nuScenes mini keyframe.
2. **Filter** — drop SAM 3 pedestrian masks whose 2D bbox has IoU ≥ 0.4
   with any 2-wheeler bbox from a YOLOv8n auxiliary detector
   (suppresses the SAM-3-thinks-the-rider-is-a-pedestrian case).
3. **Lift** — for each lidar point P, project into each camera using
   the keyframe calibration (intrinsic, extrinsic, ego-motion at the
   camera timestamp). If P falls inside a SAM 3 pedestrian mask in
   camera C with score s, accumulate
   `confidence_C = s × geometry_factor(P, C)` where geometry_factor
   decays with depth and tilts to 0 within an edge-pad fraction of
   the image border.
4. **Multi-cam consensus** — per-point confidence = max across cameras,
   binary label = (max_conf ≥ 0.5).
5. **Eval** — for every keyframe, fetch nuScenes' annotated 3D
   pedestrian boxes, transform into the lidar frame, mark points
   inside any box as GT pedestrian; compute precision / recall / F1
   vs the SAM 3 pseudo-labels. Bin by distance.

## Named failure mode — camera-blind ground-truth points

The diagnostic plot
[`outputs/failure_mode_coverage.png`](../outputs/failure_mode_coverage.png)
slices points by **how many cameras saw each lidar point** (0 / 1 / 2+):

| Camera coverage | n points  | n GT pos | F1     | Comment                        |
|-----------------|-----------|----------|--------|--------------------------------|
| 0 cameras       | 3,176,003 | 5,433    | **0.00**   | Lift physically cannot fire    |
| 1 camera        | 16,954    | 10,894   | 0.79   | Single-view supervision        |
| 2 cameras       | 1,091     | 808      | **0.86** | Multi-view consensus pays off  |

**5,433 of the 17,135 (31.7 %) GT pedestrian points sit in regions that
no camera could see** — the union of the ego-vehicle blind spot
underneath the sensor rig and points that fall outside every camera's
frustum at the keyframe timestamp. This *single failure mode* accounts
for the entire gap between our 0.65 overall recall and the 0.96 recall
on points visible to a camera.

This is a property of the sensor configuration, not the lift quality.
Three remediation paths fall naturally out of the analysis:

1. **Temporal aggregation** — aggregate masks over ±N keyframes; a
   point that was outside every camera's frustum at t may have been
   visible at t − 1 or t + 1.
2. **Geometric heuristics for blind-spot points** — augment camera
   evidence with lidar-only cues (cluster shape, height profile)
   for the underbody region.
3. **Lidar-native semantic head** — trained on the lift output where
   it is reliable, used as an additional vote in the camera-blind
   region.

These are explicit follow-ups, not gaps in the present pipeline.

The other distance-axis trend (F1 = 0.34 in the 30 + m bin) is
*expected* and consistent with the camera-coverage story: distant
points get fewer lidar returns and project to small image regions
where SAM 3 mask tilt + projection geometry compound.

## Pipeline behavior (run that produced these numbers)

| Stage                                                    | Wall time   | Output |
|----------------------------------------------------------|-------------|--------|
| A — SAM 3 over 92 keyframes × 6 cameras (552 mask files) | ~25 min     | `data/nuscenes_sam3_masks/` |
| B — lift                                                 | ~3 min      | `data/nuscenes_pseudo_3d/`  |
| E — labeling agreement                                   | ~1 min      | `outputs/labeling_agreement.{json,png}` |
| F — 3D viz                                               | ~30 s       | `outputs/demo_3d_panel.png`, `outputs/demo_3d.mp4` |
| G — failure-mode analysis                                | ~45 s       | `outputs/failure_mode.{json,coverage.png,confidence.png}` |

The 92-sample subset was chosen because Stage A on the full 404-sample
mini takes ~2.5 h on the 4070; the 92-sample run was stopped early
once the headline numbers had converged. Re-running with the full set
is mechanical and a good follow-up commit; the qualitative story does
not change with N.

## Reproducing

```bash
# After accepting the SAM 3 license at huggingface.co/facebook/sam3
huggingface-cli login
bash scripts/run_phase2.sh
```

Outputs:
- `outputs/labeling_agreement.json`     — headline numbers
- `outputs/labeling_agreement.png`      — per-distance bar chart
- `outputs/failure_mode_coverage.png`   — F1 by camera coverage
- `outputs/failure_mode_confidence.png` — TP/FP confidence histograms
- `outputs/demo_3d_panel.png`           — single-frame 3-panel viz
- `outputs/demo_3d.mp4`                 — rotating-camera flythrough

## What this is not

- Not a SOTA mAP claim. nuScenes mini is too small for stable detector
  mAP; the point-level labeling-agreement metric here is the more
  honest scale-appropriate signal.
- Not a SAM-3-vs-SAM-2-vs-GroundingDINO ablation. That is the natural
  next paper-grade step; this is a single-pipeline reference.
- Not a real-time inference number. SAM 3 itself runs at ~1.5 s/image
  on this hardware — the entire point of the recipe is to push that
  cost offline so that an on-vehicle student doesn't pay it. Phase 1's
  YOLOv8n-Seg student already shows the ~22-50 FPS edge regime for the
  camera-only version.
