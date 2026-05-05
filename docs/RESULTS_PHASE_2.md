# Phase 2 Results — SAM 3-supervised lidar pseudo-labels on nuScenes mini

> Numbers below are from a single end-to-end run on an RTX 4070 Laptop
> (8 GB VRAM, FP16) with no human 3D box labels used during training.
> The "training" in this phase is the *pseudo-labeling pipeline itself*;
> the downstream PointPillars detector is a follow-up.

## TL;DR

We treat SAM 3 as a 2D oracle, run it on each of the 6 nuScenes
cameras per keyframe, and project the resulting pedestrian masks onto
the lidar sweep with multi-camera consensus. The result is a per-point
binary pedestrian label whose **point-level agreement with nuScenes
3D ground-truth pedestrian boxes** is the primary metric.

| Metric (point-level vs nuScenes GT)   | Value         |
|---------------------------------------|---------------|
| Overall precision                     | _(filled by run)_ |
| Overall recall                        | _(filled by run)_ |
| Overall F1                            | _(filled by run)_ |
| F1 @ 0-15 m                           | _(filled)_    |
| F1 @ 15-30 m                          | _(filled)_    |
| F1 @ 30+ m                            | _(filled)_    |
| n samples evaluated                   | _(filled)_    |
| n total lidar points                  | _(filled)_    |
| n GT pedestrian points                | _(filled)_    |
| n SAM-3 pseudo-pedestrian points      | _(filled)_    |

(Full breakdown in `outputs/labeling_agreement.json`.)

## Pipeline (recap)

1. **Oracle** — SAM 3 (`facebook/sam3`) with concept prompts
   `{pedestrian, person walking, person standing, person crossing the
   street}`, negative-prompted against `{cyclist, person on bicycle,
   person on scooter, person in wheelchair}`. Run on every camera
   image of every nuScenes mini keyframe.
2. **Filter** — drop SAM 3 pedestrian masks whose 2D bbox has IoU ≥
   0.4 with any 2-wheeler bbox from a YOLOv8n auxiliary detector.
3. **Lift** — for each lidar point P, project into each camera using
   the keyframe calibration (intrinsic, extrinsic, ego-motion at the
   camera timestamp). If P falls inside a SAM 3 pedestrian mask in
   camera C with score s, accumulate `confidence_C = s *
   geometry_factor(P, C)` where the geometry factor decays with depth
   and tilts to 0 at image edges.
4. **Multi-cam consensus** — per-point confidence = max across cameras,
   binary label = (max_conf ≥ 0.5).
5. **Eval** — for every keyframe, fetch nuScenes' annotated 3D
   pedestrian boxes, transform to lidar frame, mark points inside any
   box as GT pedestrian; compute precision/recall/F1 vs the SAM-3
   pseudo-labels. Bin by distance.

## Wall time on the run that produced these numbers

| Stage                                         | Wall time   |
|-----------------------------------------------|-------------|
| A — SAM 3 on 404 keyframes × 6 cameras        | _(filled)_  |
| B — lift                                      | _(filled)_  |
| E — labeling agreement                        | _(filled)_  |
| F — 3D viz                                    | _(filled)_  |
| **Total**                                     | _(filled)_  |

## Named failure mode

_(short paragraph after run: which distance bin / camera-coverage bin
breaks; which kind of scene; one diagnostic plot reference.)_

## Reproducing

```bash
huggingface-cli login           # accept facebook/sam3 license first
bash scripts/run_phase2.sh      # ~75 min on RTX 4070 Laptop, FP16
```

Outputs land in:

- `outputs/labeling_agreement.json`     headline numbers
- `outputs/labeling_agreement.png`      per-distance bar chart
- `outputs/demo_3d_panel.png`           single-frame 3-panel viz
- `outputs/demo_3d.mp4`                 rotating-camera flythrough

## What this is not

- Not a SOTA mAP claim. nuScenes mini is too small for stable mAP at
  the detector level. Stage E's labeling-agreement metric is the
  point-level signal that *is* meaningful at this scale.
- Not a SAM-3-vs-other-oracle ablation. That is the natural follow-up
  paper; this is a single-pipeline reference implementation.
- Not real-time. Stage A is 1.5 s/image of teacher inference; the
  whole point is to push that cost offline so an on-vehicle student
  doesn't pay it. Phase 1's YOLOv8n-Seg student already showed the
  ~22-50 FPS edge regime for the camera-only version.
