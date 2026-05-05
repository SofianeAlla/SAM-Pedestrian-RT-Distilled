# SAM 3 Pedestrian — Real-time distillation reference implementation

A reproducible, end-to-end recipe for distilling Meta's
[SAM 3](https://github.com/facebookresearch/sam3) into both a
**real-time on-device 2D segmenter** and a **lidar-native 3D
pedestrian supervisor** for AV-style perception. Built and verified
on a single consumer NVIDIA laptop GPU (RTX 4070, 8 GB).

This repository is a **public reference implementation** for engineers
wiring SAM 3 (or any concept-prompted 2D foundation model) into their
own perception data pipelines. The pedestrian class is treated as
*expert #1* of a future Mixture-of-Experts perception stack — each
subsequent class (vehicle, cyclist, sign, lane, drivable area, long-tail)
clones the same recipe, so adding experts does not scale runtime cost.
EMC2 (ICCV 2025) is the closest published prior art for the
scenario-aware MoE-on-edge pattern.

## Headline results

### Phase 2 — SAM 3 as 2D oracle for 3D pedestrian supervision (nuScenes mini)

Point-level agreement of SAM-3-derived lidar pseudo-labels vs nuScenes
3D pedestrian box ground truth, **full nuScenes mini (404 keyframes,
14.0 M lidar points, 47,965 GT pedestrian points), zero human 3D
labels used during pseudo-labeling**:

| | precision | recall | F1 |
|---|:-:|:-:|:-:|
| **Overall** | **0.662** | **0.670** | **0.666** |
| 0–15 m  | 0.82 | 0.71 | 0.76 |
| 15–30 m | 0.46 | 0.63 | 0.53 |
| 30+ m   | 0.19 | 0.32 | 0.24 |

**Named failure mode** — by camera coverage (how many of 6 cameras saw each lidar point):

| coverage | n points    | F1   | comment |
|---|---:|:-:|---|
| 0 cams | 13,973,315 | 0.00 | lift physically cannot fire — sensor blind spot |
| 1 cam  |     50,056 | 0.78 | single-view supervision |
| 2 cams |      2,837 | 0.84 | multi-view consensus pays off |

**14,803 of 47,965 GT pedestrian points (30.9 %) sit in the union of
the under-vehicle blind spot and points outside every camera frustum
at the keyframe instant.** The lift cannot fire on those by
construction — this single sensor-config story accounts for the
entire gap between 0.67 overall recall and 0.94 recall on
camera-visible points.

Full writeup: [`docs/RESULTS_PHASE_2.md`](docs/RESULTS_PHASE_2.md).
Plan: [`docs/PHASE_2_PLAN.md`](docs/PHASE_2_PLAN.md).
Run end-to-end with `bash scripts/run_phase2.sh`.

### Phase 1 — SAM 3 → YOLOv8n-Seg edge student (Intel sample-videos)

Real-time 2D camera-only pedestrian detection + segmentation on the
held-out Intel `person-bicycle-car-detection.mp4` clip, 647 frames,
RTX 4070 FP16, conf=0.25:

| Demo | Weights / Training | Mean inf | FPS | Ped detections |
|---|---|---:|---:|---:|
| `demo_baseline.mp4` | `yolov8n-seg.pt` — COCO out-of-the-box | 16.5 ms | 50.5 | 193 / 647 |
| `demo_smoke.mp4` | 35 SAM 3 labels, `single_cls=True` (head reset) | 15.4 ms | 56.0 | 5 / 647 |
| `demo_1h.mp4` | **500 COCO val2017 person images SAM-3-relabeled, 38 epochs frozen backbone, AdamW lr0=0.001, 80 COCO names retained** | 37.7 ms | 22.0 | **233 / 647** |

`demo_1h.mp4` finds **+21 % more pedestrians than the COCO baseline
on a video the student has never seen** (data hygiene: train on COCO,
demo on Intel — zero overlap). Full Phase 1 detail in the section
below.

## Artifacts on this repo

Phase 2:
- [`outputs/demo_3d.mp4`](outputs/demo_3d.mp4) — rotating-BEV flythrough on a real keyframe
- [`outputs/demo_3d_panel.png`](outputs/demo_3d_panel.png) — front cam + SAM 3 mask + lidar BEV with pseudo-labels and GT 3D boxes
- [`outputs/labeling_agreement.png`](outputs/labeling_agreement.png) — per-distance P/R/F1
- [`outputs/failure_mode_coverage.png`](outputs/failure_mode_coverage.png) — F1 by camera-coverage bin (the named failure mode)
- [`outputs/failure_mode_confidence.png`](outputs/failure_mode_confidence.png) — TP/FP confidence histograms
- [`outputs/labeling_agreement.json`](outputs/labeling_agreement.json), [`outputs/failure_mode.json`](outputs/failure_mode.json) — raw numbers

Phase 1:
- [`outputs/demo_baseline.mp4`](outputs/demo_baseline.mp4), [`outputs/demo_smoke.mp4`](outputs/demo_smoke.mp4), [`outputs/demo_1h.mp4`](outputs/demo_1h.mp4)
- Training run artifacts under [`runs/segment/runs/ped_smoke/`](runs/segment/runs/ped_smoke/) and [`runs/segment/runs/ped_1h/`](runs/segment/runs/ped_1h/)

## What's in here

```
docs/
  PHASE_2_PLAN.md              3D supervision via 2D oracle on nuScenes
  RESULTS_PHASE_2.md           Numbers + named failure mode

# Phase 1 — 2D camera distillation
teacher/
  sam3_autolabel.py            SAM 3 → YOLO-Seg pseudo-labels
  pseudolabel_filter.py        Drop pedestrian-on-bicycle false positives
  carla_synth.py               CARLA pedestrian capture (optional)
distill/
  configs/ped_yolov8n.yaml     Training config
  train.py                     YOLOv8n-Seg fine-tune on SAM 3 labels
runtime/
  pedestrian_expert.py         Inference wrapper + Expert plugin contract
  export_onnx.py               PyTorch → ONNX
  build_trt.py                 ONNX → desktop TensorRT engine
  demo_live.py                 Video / webcam → annotated MP4
eval/
  benchmark.py                 Latency + throughput on local GPU
scripts/
  setup.sh                     Install + sanity check
  fetch_sample_video.py        Pull a CC sample driving video
  build_coco_person_subset.py  COCO val2017 → person-only training subset
  build_distill_dataset.py     Two SAM 3 outputs → canonical YOLO dataset
  run_smoke.sh                 Tiny end-to-end run (~30-60 min)
  run_1h_v2.sh                 Diversified ~1 h training run
  run_full.sh                  Overnight full corpus + epochs

# Phase 2 — SAM 3 as 2D oracle for 3D pedestrian supervision
teacher/
  nuscenes_sam3_oracle.py      SAM 3 over the 6 nuScenes cameras (Stage A)
  lidar_lift.py                Project masks onto lidar points (Stage B)
eval/
  labeling_agreement.py        Point-level P/R/F1 vs nuScenes GT (Stage E)
  viz_3d.py                    3-panel viz + rotating BEV flythrough (Stage F)
  failure_mode.py              By-camera-coverage + confidence breakdown (Stage G)
scripts/
  fetch_nuscenes_mini.py       One-time nuScenes mini downloader (~4 GB)
  run_phase2.sh                End-to-end Phase 2 orchestrator
```

## Setup

```bash
bash scripts/setup.sh
huggingface-cli login          # accept facebook/sam3 license first
```

SAM 3 weights are gated by Meta's license. Visit
<https://huggingface.co/facebook/sam3> and click "Access repository"
before logging in.

## Quick demo (no training needed)

```bash
PYTHONPATH=$PWD python -m runtime.demo_live \
  --weights yolov8n-seg.pt \
  --video data/sample_videos/person-bicycle-car-detection.mp4 \
  --output outputs/demo_baseline.mp4
```

Produces a working pedestrian video using YOLOv8n-Seg's COCO-pretrained
"person" class — the reference baseline before any distillation.

## Phase 2 end-to-end (3D pseudo-labels on nuScenes mini)

```bash
huggingface-cli login                  # one-time
bash scripts/run_phase2.sh             # ~75 min on RTX 4070 (subset),
                                       # ~3 h for full nuScenes mini
```

Stages, all driven by the orchestrator:

```
A. SAM 3 over the 6 nuScenes cameras of every keyframe
B. Lidar-camera projection + multi-view consensus → per-point labels
E. Point-level precision/recall/F1 vs nuScenes 3D pedestrian box GT
F. 3-panel viz + 13 MB rotating-BEV flythrough video
G. Failure-mode plot: F1 by camera coverage + confidence histograms
```

## Phase 1 end-to-end (2D camera distillation)

```bash
# Smoke (a few hundred images, ~30-60 min)
bash scripts/run_smoke.sh

# Diversified 1 h run on COCO val2017 person subset (the demo_1h.mp4
# numbers above)
bash scripts/run_1h_v2.sh

# Overnight full corpus + epochs
bash scripts/run_full.sh
```

The Phase 1 trainer initializes from `yolov8n-seg.pt` (COCO-pretrained)
and writes the SAM-3-finetuned student to a `runs/.../weights/best.pt`,
plus an annotated demo MP4 of the student running on a held-out video.

## Phase 1 details — validation metrics on held-out COCO split (50 images, never seen during training)

| Metric         | Box   | Mask  |
|----------------|:-----:|:-----:|
| Precision      | 0.83  | 0.82  |
| Recall         | 0.70  | 0.74  |
| mAP@0.5        | 0.73  | 0.77  |
| mAP@0.5:0.95   | 0.53  | 0.52  |

**Data hygiene matters.** The `demo_1h.mp4` student was trained on
COCO val2017 person images and demoed on Intel sample-videos — zero
overlap between training, validation, and the demo footage. The
`demo_smoke.mp4` is kept as historical context: trained on 35 SAM 3
pseudo-labels with `single_cls=True`, which resets the YOLOv8n-Seg
head and destroys COCO's pretrained person prior; included so the
v1→v2 fix is visible in the commit history.

**Latency note.** The 1 h run is slower per frame than the smoke run
(37.7 ms vs 15.4 ms) because its head retains all 80 COCO classes —
more raw NMS candidates per anchor at the same confidence threshold.
Still real-time at 22 FPS on a single laptop GPU. Collapsing back to
a single-class head post-distillation, or raising `conf`, recovers
the 50+ FPS regime trivially.

## Architectural commitments worth keeping

1. **Backbone outputs feature pyramid at named taps** (`P3/P4/P5`). The
   `Expert` Protocol in `runtime/pedestrian_expert.py` makes these the
   integration surface for future experts. Don't fuse class-specific
   computation into the backbone.

2. **Single class today, copy-paste tomorrow.** Adding cyclists,
   vehicles, signs is a new head + a new SAM 3 prompt set + a new
   YAML — the training pipeline doesn't change.

3. **Negative prompts matter.** The pedestrian expert is explicitly
   negative-prompted against cyclists and scooter riders so its
   training set doesn't pollute the future cyclist expert. The
   post-hoc filter in `teacher/pseudolabel_filter.py`
   belt-and-suspenders this.

4. **Use the foundation model offline as a teacher, not online as an
   oracle.** SAM 3 itself runs at ~1.5 s/image on this hardware — the
   point of the whole recipe is to push that cost offline so an
   on-vehicle or on-edge student doesn't pay it at inference time.

## Out of scope (intentional, named follow-ups)

- **PointPillars 3D detector** distilled on the Phase 2 pseudo-labels.
  Phase 2 currently produces the supervision; the lidar-native student
  is the natural next phase.
- **Temporal aggregation** across ±N keyframes to recover the
  camera-blind GT points (the named failure mode above).
- **SAM 3 vs SAM 2 vs GroundingDINO+SAM ablation.** This is the
  natural paper-grade follow-up; the present repo is intentionally a
  single-pipeline reference, not a comparison study.
- **Orin AGX port.** Same code paths apply; needs the device for INT8
  calibration and `trtexec` on the Jetson.
- **MoE router and second expert** (vehicle / cyclist / sign).
- **Multi-camera batching, sensor fusion, tracking.**

## References

- SAM 3: <https://github.com/facebookresearch/sam3>
- nuScenes: <https://www.nuscenes.org>
- Ultralytics YOLOv8: <https://docs.ultralytics.com/>
- EMC2 — scenario-aware MoE for edge AV, ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Towards_Accurate_and_Efficient_3D_Object_Detection_for_Autonomous_Driving_ICCV_2025_paper.pdf>
- Adjacent prior art on foundation-model-supervised 3D perception:
  Seal (CVPR 2024), OpenMask3D, OV-3DET, SAMesh.
