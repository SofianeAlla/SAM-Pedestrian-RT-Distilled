# Phase 2 — SAM 3 as 2D oracle for 3D pedestrian supervision (nuScenes mini)

## Why this phase

Phase 1 shipped a 2D camera-only pedestrian segmenter distilled from
SAM 3, real-time on the RTX 4070 Laptop. For a 3D ML conversation
(specifically: Zoox application), 2D camera output is not the
deliverable — Zoox runs on point clouds. Phase 2 turns Phase 1 from a
camera demo into a *3D* deliverable by treating SAM 3 as a 2D oracle
that bootstraps supervision for a lidar-native student.

This phase is intentionally scoped to a **single end-to-end run with
one set of numbers + a 3D viz**, not a paper-grade ablation. It is the
laptop-feasible version of the SAM-as-2D-oracle direction (SAMesh,
Seal, OpenMask3D, OV-3DET in spirit).

## Locked decisions

- **Dataset**: nuScenes **mini** (10 scenes, ~400 keyframes, 6 cams +
  lidar, 4 GB on disk). Official `nuscenes-devkit` for I/O and eval.
  No KITTI in this round (cross-dataset eval is a follow-up).
- **Compute**: laptop only — RTX 4070 Laptop, 8 GB VRAM, FP16.
- **Project root**: `C:\dev\SAM-Pedestrian-RT-Distilled\`.
  Off OneDrive to avoid sync conflicts on intermediate checkpoints.
- **Out**: ablation tables (no SAM 3 vs SAM 2 vs GroundingDINO+SAM
  comparisons, no aug/LR sweeps, no fine-tune-with-GT row).
  One pipeline, one model, one set of numbers.

## Pipeline

### Stage A — SAM 3 over the 6 nuScenes cameras (oracle)
For each keyframe:
1. Load all 6 camera images (CAM_FRONT, CAM_FRONT_LEFT,
   CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT).
2. Run SAM 3 with person-on-foot concept prompts (positive set as in
   Phase 1; negative-prompted against cyclist/scooter/wheelchair).
3. Cache per-camera binary mask + per-instance score to disk.
4. Reuse `teacher.pseudolabel_filter` to drop pedestrian-on-bicycle
   false positives per camera.

Compute budget: ~400 keyframes × 6 cams × ~1.5 s = **~60 min** total
on the 4070, one-time cost. Cached so subsequent training is free.

### Stage B — 2D→3D lift via camera-lidar projection
For each keyframe lidar sweep:
1. For each lidar point P, project into each of the 6 camera frames
   using `nuscenes-devkit` calibration (intrinsic + extrinsic +
   ego-motion compensation).
2. If P projects inside a SAM 3 pedestrian mask in camera C with
   score s, accumulate a soft label
   `confidence_C = s * geometry_factor(P, C)` where the geometry
   factor decays with depth and tilts to 0 at image edges.
3. Multi-camera consensus: a point is labeled pedestrian iff at least
   one camera's confidence exceeds `tau` (default 0.5). Final
   per-point confidence = max across cameras.
4. Optional temporal aggregation: ±2 sweeps (so a partially-seen
   pedestrian in a single sweep gets reinforced by neighbors). Off by
   default in this phase to keep the pipeline tight.

Output: per-keyframe lidar tensor `(N, 4)` + per-point label tensor
`(N,)` with values `{0=non-ped, 1=pedestrian}` and a confidence
`(N,)` in `[0, 1]`. Stored in `data/nuscenes_pseudo_3d/`.

### Stage C — PointPillars 3D pedestrian detector (student)
- Architecture: PointPillars (voxel CNN), single-class pedestrian
  head. ~3-5 M params. Use the standard `mmdet3d` or `OpenPCDet`
  config as a baseline; minimal modifications.
- Training: SAM-3-derived pseudo-labels (Stage B output) as
  supervision. Single training config (no sweeps).
- Hardware budget: 50 epochs on nuScenes mini, batch 4, ~3-4 hours
  on the 4070.

### Stage D — Eval
Single set of numbers, official nuScenes eval (`nuscenes-devkit`
`evaluate.py`):
- **Pedestrian mAP** at 0.5 / 1.0 / 2.0 / 4.0 m thresholds.
- **NDS** (nuScenes detection score) — the official aggregate metric.
- Per-distance recall: 0-15 m, 15-30 m, 30+ m. Free side-product of
  the eval; a sentence-long failure-mode story for the interview.

### Stage E — 3D viz (the conversation starter)
- Render one held-out keyframe with:
  - Lidar points colored by class (gray = non-ped, orange = ped).
  - Predicted 3D bounding boxes from PointPillars.
  - Optional: SAM 3 camera mask overlaid on the same keyframe's
    front camera, side-by-side.
- Save as a 10-15 s video sweeping the camera around the scene.
- Path: `outputs/demo_3d.mp4`. This is the artifact you open the
  Zoox conversation with.

## Implementation surface (additive on Phase 1)

```
docs/
  PHASE_2_PLAN.md                  this file
data/
  nuscenes/                        nuScenes mini extracted (gitignored)
  nuscenes_sam3_masks/             SAM 3 mask cache (gitignored)
  nuscenes_pseudo_3d/              per-keyframe lidar pseudo-labels (gitignored)
teacher/
  nuscenes_sam3_oracle.py          Stage A: SAM 3 over all 6 cams
  lidar_lift.py                    Stage B: 2D mask -> 3D point labels
runtime/
  pointpillars_expert.py           Stage C/D student wrapper, mirrors
                                   PedestrianExpert API for the MoE
                                   `Expert` Protocol
distill/
  configs/ped_pointpillars.yaml    Stage C training config
  train_3d.py                      Stage C training driver
eval/
  nuscenes_eval.py                 Stage D official nuScenes eval
  viz_3d.py                        Stage E lidar + box rendering
scripts/
  fetch_nuscenes_mini.py           one-time download helper
  run_phase2.sh                    end-to-end orchestrator
```

External deps to add:
- `nuscenes-devkit` (official I/O + eval)
- `mmdet3d` *or* `OpenPCDet` (PointPillars implementation; pick one
  based on Windows installability — `OpenPCDet` is usually friendlier)
- `open3d` (Stage E viz)

## Single end-to-end run plan

Approximate wall time on the RTX 4070 Laptop, 8 GB:
- Stage A (SAM 3 over 2,400 images):     ~60 min
- Stage B (lift + cache):                ~10 min
- Stage C (PointPillars 50 epochs):      ~3-4 h
- Stage D (eval):                        ~5 min
- Stage E (viz render):                  ~5 min
- **Total: ~5 hours**, suitable for one focused session.

## What this is NOT

- Not a paper. No ablation table.
- Not SOTA. nuScenes mini is too small; the headline number is "did
  the SAM-3-as-3D-oracle pipeline produce a non-trivial nuScenes
  pedestrian mAP from a laptop run with no human 3D labels?"
- Not multi-class, not multi-frame tracking, not fusion. Pedestrian
  only, single-frame, lidar-only student, 6-cam supervision.

## Conversation framing for Zoox

Open with the v1→v2→Phase 2 progression as a *learning* artifact:

> "I built a SAM 3 → 2D camera segmenter, caught my own data leakage
> in v1, fixed it in v2, then realized 2D wasn't the right deliverable
> for 3D ML and pivoted to using SAM 3 as a 2D oracle that supervises
> a PointPillars student on nuScenes lidar. Here's the 3D output and
> the nuScenes mAP. Now ask me what's wrong with my approach."

The mistakes (leakage, head reset, latency regression) are part of
the story. Strong candidates make and catch their own errors visibly.
