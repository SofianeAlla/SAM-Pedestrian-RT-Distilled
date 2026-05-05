# Phase 2 — SAM 3 as a 2D oracle for 3D pedestrian supervision (nuScenes mini)

## Why this phase

Phase 1 shipped a 2D camera-only pedestrian detector + segmenter,
distilled from SAM 3, real-time on consumer GPU hardware. For an
autonomous-driving perception stack, the deliverable is *3D* — point
clouds with 3D bounding boxes, not image-space masks. Phase 2 turns
the Phase 1 pipeline into a 3D one by treating SAM 3 as a **2D oracle
that bootstraps supervision for a lidar-native student**, with no
human-labeled 3D boxes used in training.

This is a concrete, end-to-end recipe for the open question: *can a
2D foundation model with text prompting carry enough scene
understanding to supervise 3D pedestrian detection in a self-driving
setting?* The answer matters because human 3D box labels are the
single most expensive line item in AV perception data pipelines.

The architectural pattern is in the same family as
[SAMesh](https://arxiv.org/abs/2407.16692) (object-centric SAM-to-mesh),
[Seal](https://arxiv.org/abs/2306.09347) (CVPR 2024 — SAM-supervised
3D point cloud features), [OpenMask3D](https://arxiv.org/abs/2306.13631)
and [OV-3DET](https://arxiv.org/abs/2304.00788) — but specialized to:
- the AV outdoor-scene regime (sparse, dynamic, multi-view lidar);
- using **SAM 3** (Nov 2025) rather than SAM 1/2 + GroundingDINO,
  which simplifies the prompt-→-mask path to a single concept-prompted
  call;
- a single fully-laptop-runnable training run, no cloud GPUs.

## Locked decisions

- **Dataset**: nuScenes **mini** (10 scenes, ~400 keyframes, 6 cams +
  lidar, 4 GB on disk). Official `nuscenes-devkit` for I/O and eval.
- **Compute**: laptop only — RTX 4070 Laptop, 8 GB VRAM, FP16.
- **Project root**: `C:\dev\SAM-Pedestrian-RT-Distilled\` (off
  OneDrive to avoid sync conflicts on intermediate checkpoints).
- **Out of scope for this phase**: ablation tables (no SAM 3 vs SAM 2
  vs GroundingDINO+SAM rows, no aug/LR sweeps, no fine-tune-with-GT
  comparisons). One pipeline, one model, one set of numbers + one
  named failure mode. Cross-dataset eval, tracking, multi-class,
  fusion, and the MoE router are all explicit follow-ups.

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

Compute budget: ~400 keyframes × 6 cams × ~1.5 s ≈ **~60 min** total
on the 4070, one-time cost. Cached so subsequent training is free.

### Stage B — 2D→3D lift via camera-lidar projection
For each keyframe lidar sweep:
1. For each lidar point P, project into each of the 6 camera frames
   using `nuscenes-devkit` calibration (intrinsic + extrinsic +
   ego-motion compensation between sweep and image timestamps).
2. If P projects inside a SAM 3 pedestrian mask in camera C with
   score s, accumulate a soft label
   `confidence_C = s * geometry_factor(P, C)` where the geometry
   factor decays with depth and tilts to 0 at image edges.
3. Multi-camera consensus: a point is labeled pedestrian iff at least
   one camera's confidence exceeds `tau` (default 0.5). Final
   per-point confidence = max across cameras.

Output: per-keyframe lidar tensor `(N, 4)` + per-point label tensor
`(N,)` with values `{0=non-ped, 1=pedestrian}` and a confidence
`(N,)` in `[0, 1]`. Stored in `data/nuscenes_pseudo_3d/`.

### Stage C — PointPillars 3D pedestrian detector (student)
- Architecture: PointPillars (voxel CNN), single-class pedestrian
  head. ~3-5 M params. Standard config from `OpenPCDet` or `mmdet3d`.
- Training: SAM-3-derived pseudo-labels (Stage B output) as
  supervision. Single training config — no sweeps.
- Hardware budget: ~50 epochs on nuScenes mini, batch 4, ~3-4 h on
  the 4070.
- Mirrors the `Expert` Protocol from `runtime/pedestrian_expert.py`
  so the 3D student plugs into the same future-MoE substrate as the
  2D Phase 1 student.

### Stage D — Eval (one set of numbers)
Single set of headline numbers via the official `nuscenes-devkit`
eval pipeline:
- **Pedestrian mAP** at 0.5 / 1.0 / 2.0 / 4.0 m thresholds.
- **NDS** (nuScenes detection score) — the official aggregate metric.
- **Per-distance recall**: 0-15 m / 15-30 m / 30+ m, free side-product
  of the eval run; useful for the failure-mode analysis below.

### Stage E — Labeling-agreement check (no detector required)
A second, complementary number that tests the *lift itself*, not the
downstream detector:
- For each held-out nuScenes keyframe, project the dataset's 3D
  pedestrian boxes into the lidar sweep to get per-point GT labels.
- Compare against the Stage B SAM-3-derived per-point labels.
- Report point-level precision, recall, F1.
- This metric is meaningful at mini-scale even if Stage C's mAP isn't
  (mini is too small for a stable mAP), so it stands alone if Stage C
  is skipped.

### Stage F — 3D viz
- Render one held-out keyframe with:
  - Lidar points colored by class (gray = non-ped, orange = ped).
  - Predicted 3D bounding boxes from PointPillars.
  - SAM 3 camera mask overlaid on the same keyframe's front camera,
    side-by-side panel.
- Save as a 10-15 s rotating-camera flythrough.
- Path: `outputs/demo_3d.mp4`.

### Stage G — Named failure mode
Pick one specific, diagnosable failure of the lift on the held-out
set. Examples we expect to find:
- Heavily-occluded pedestrians where 4+ cameras see only the
  occluder, multi-cam consensus misses.
- Calibration-drift bias in long-range projections beyond ~25 m.
- Blind-spot pedestrians under the ego vehicle that no camera sees.
Document with a diagnostic plot in `docs/FAILURE_MODES.md`.

## Implementation surface (additive on Phase 1)

```
docs/
  PHASE_2_PLAN.md                  this file
  RESULTS_PHASE_2.md               numbers + failure mode (lands when Stage E/F done)
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
  labeling_agreement.py            Stage E lift-only metric
  viz_3d.py                        Stage F lidar + box rendering
scripts/
  fetch_nuscenes_mini.py           one-time download helper
  run_phase2.sh                    end-to-end orchestrator
```

External deps to add:
- `nuscenes-devkit` — official I/O + eval
- `OpenPCDet` *or* `mmdet3d` — PointPillars implementation
- `open3d` — Stage F viz

## End-to-end run plan

Approximate wall time on the RTX 4070 Laptop, 8 GB:

| Stage                                          | Wall time     |
|------------------------------------------------|---------------|
| A (SAM 3 over ~2,400 keyframe-camera images)   | ~60 min       |
| B (lift + cache)                               | ~10 min       |
| E (labeling agreement, runs on B output)       | ~5 min        |
| C (PointPillars 50 epochs)                     | ~3-4 h        |
| D (nuScenes eval)                              | ~5 min        |
| F (viz render)                                 | ~5 min        |
| G (failure-mode diagnostic + plot)             | ~30 min       |
| **Total end-to-end**                           | **~5 h**      |

Stage E gives us a meaningful number even if C is delayed, so it is
the minimum publishable cut.

## What this is NOT

- Not a SOTA paper. nuScenes mini is too small for a stable headline
  mAP; Stage E's labeling-agreement metric is the more honest one at
  this scale.
- Not multi-class, not multi-frame tracking, not camera+lidar fusion
  in the student. Pedestrian only, single-frame, lidar-only student,
  6-camera supervision.
- Not a SAM-3-vs-SAM-2 ablation. That's the natural next-step paper.

## Why a public reference implementation

Foundation-model-supervised 3D perception is an active research area
but most work targets either (a) static indoor scenes (OpenMask3D
class) or (b) generic 3D semantic segmentation with SAM 1/2 (Seal
class). A reproducible **AV-pedestrian, SAM 3, laptop-runnable**
recipe doesn't exist publicly at the time of this writing. The repo
is meant to fill that gap for engineers who want to wire SAM 3 into
their own perception data pipelines without reading 8 papers first.
