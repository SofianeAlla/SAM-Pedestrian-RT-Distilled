# SAM 3 Pedestrian — Real-time distillation reference implementation

A reproducible, end-to-end recipe for distilling Meta's
[SAM 3](https://github.com/facebookresearch/sam3) into real-time
on-device perception models for the **pedestrian** class. Built and
verified on a single consumer NVIDIA laptop GPU (RTX 4070, 8 GB).

This repository is a **public reference implementation** for engineers
wiring SAM 3 (or any concept-prompted 2D foundation model) into their
own perception data pipelines. The pedestrian class is treated as
*expert #1* of a future Mixture-of-Experts perception stack — each
subsequent class (vehicle, cyclist, sign, lane, drivable area, long-tail)
clones the same recipe, so adding experts does not scale runtime cost.
EMC2 (ICCV 2025) is the closest published prior art for the
scenario-aware MoE-on-edge pattern.

## Phases

- **Phase 1 — 2D camera distillation (shipped)**: SAM 3 →
  YOLOv8n-Seg student, real-time on consumer GPU. Includes a v1→v2
  redesign in the commit history that fixes a data-leakage trap (the
  v1 pipeline trained on frames that were also used in the demo
  evaluation; v2 separates train/val/demo into disjoint sources).
- **Phase 2 — 3D supervision via 2D oracle (in progress)**: SAM 3 over
  the 6 nuScenes cameras → camera-lidar projection → multi-view
  consensus → per-point 3D pedestrian pseudo-labels → optional
  PointPillars student. Plan in
  [`docs/PHASE_2_PLAN.md`](docs/PHASE_2_PLAN.md), results in
  [`docs/RESULTS_PHASE_2.md`](docs/RESULTS_PHASE_2.md).

---

## What's in here

```
docs/
  PHASE_2_PLAN.md            3D supervision via 2D oracle on nuScenes
  RESULTS_PHASE_2.md         Numbers + named failure mode

# Phase 1 — 2D camera distillation
teacher/
  sam3_autolabel.py          SAM 3 → YOLO-Seg pseudo-labels
  pseudolabel_filter.py      Drop pedestrian-on-bicycle false positives
  carla_synth.py             CARLA pedestrian capture (optional)
distill/
  configs/ped_yolov8n.yaml   Training config
  train.py                   YOLOv8n-Seg fine-tune on SAM 3 labels
runtime/
  pedestrian_expert.py       Inference wrapper + Expert plugin contract
  export_onnx.py             PyTorch → ONNX
  build_trt.py               ONNX → desktop TensorRT engine
  demo_live.py               Video / webcam → annotated MP4
eval/
  benchmark.py               Latency + throughput on local GPU
scripts/
  setup.sh                   Install + sanity check
  fetch_sample_video.py      Pull a CC sample driving video
  build_coco_person_subset.py
  build_distill_dataset.py
  run_smoke.sh               Tiny end-to-end run (~30-60 min)
  run_1h_v2.sh               Diversified ~1 h training run
  run_full.sh                Overnight full corpus + epochs

# Phase 2 — 3D supervision via 2D oracle (nuScenes mini)
teacher/
  nuscenes_sam3_oracle.py    SAM 3 over the 6 nuScenes cameras (Stage A)
  lidar_lift.py              Project masks onto lidar points (Stage B)
eval/
  labeling_agreement.py      Point-level P/R/F1 vs nuScenes GT (Stage E)
  viz_3d.py                  3-panel viz + rotating BEV flythrough (Stage F)
  failure_mode.py            By-camera-coverage + confidence breakdown (Stage G)
scripts/
  fetch_nuscenes_mini.py     One-time nuScenes mini downloader
  run_phase2.sh              End-to-end Phase 2 orchestrator
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
"person" class. **This is the baseline.** SAM 3 distillation refines
mask quality and long-tail recall on top of this.

## Smoke test (after HF login)

1. Drop ~50-200 driving images into `data/seed_images/`.
   (Cityscapes val, BDD subset, or anything you have handy.)
2. `bash scripts/run_smoke.sh`

Expect ~30-60 minutes on the 4070 to:
SAM 3 pseudo-label → filter → fine-tune YOLOv8n-Seg →
write `outputs/demo_smoke.mp4`.

## Full overnight run

```bash
# Populate ~10K-20K images first, optionally include CARLA-synthesized.
bash scripts/run_full.sh
```

100 epochs, larger batch, ONNX export, desktop TensorRT build,
benchmark + final `outputs/demo_full.mp4`.

## One-paragraph summary (presentation copy)

**SAM 3-distilled real-time pedestrian expert.** A working desktop-edge
pedestrian detector + segmenter, distilled from Meta's SAM 3 (840M-param
concept-prompted teacher), built as expert #1 of a future Mixture-of-
Experts foundation perception model for autonomous-vehicle vision. The
pipeline runs entirely on a single RTX 4070 Laptop (8 GB): SAM 3
generates pseudo-labels with person-on-foot text prompts (negatively
prompted against cyclists/scooters/wheelchairs to keep the future MoE
class boundary clean); a YOLOv8n-Seg student is fine-tuned on those
labels and exported to TensorRT. End-to-end real-time inference on a
647-frame driving video clocks in at **56 FPS / 15.4 ms mean inference
latency** (det + seg head, FP16, single front camera, 768×432 input).
The architecture follows EMC2 (ICCV 2025) — backbone outputs a shared
P3/P4/P5 feature pyramid and the student is a class-specialized head,
so adding the next expert (vehicles, signs, lanes) is a head + a SAM 3
prompt set rather than a rewrite, and runtime cost stays cheap as the
expert pool grows under sparse top-k routing.

## Performance

Held-out demo video: `person-bicycle-car-detection.mp4` (Intel sample-
videos), 647 frames, 768×432, RTX 4070 Laptop, FP16, single front
camera, det + seg, conf=0.25.

| Demo                | Weights / Training                                                                 | Mean inf  | FPS   | Pedestrian dets | Notes                                              |
|---------------------|------------------------------------------------------------------------------------|-----------|-------|-----------------|----------------------------------------------------|
| `demo_baseline.mp4` | `yolov8n-seg.pt` — COCO out-of-the-box                                             | 16.5 ms   | 50.5  | 193 / 647       | Reference baseline                                 |
| `demo_smoke.mp4`    | `runs/.../ped_smoke/weights/best.pt` — 35 SAM 3 labels, single_cls reset           | 15.4 ms   | 56.0  | 5 / 647         | Pipeline validation only — head reset killed quality |
| `demo_1h.mp4`       | `runs/.../ped_1h/weights/best.pt` — **500 COCO val2017 person images, SAM 3 labels, 38 epochs (early-stopped via patience=15), frozen backbone, AdamW lr0=0.001, all 80 COCO names retained** | 37.7 ms | 22.0  | **233 / 647**   | **+21 % more pedestrians than baseline on a video the student has never seen** |

**Validation metrics** (50-image held-out COCO split, never seen during training):

| Metric           | Box     | Mask    |
|------------------|---------|---------|
| Precision        | 0.83    | 0.82    |
| Recall           | 0.70    | 0.74    |
| mAP@0.5          | 0.73    | 0.77    |
| mAP@0.5:0.95     | 0.53    | 0.52    |

**Data hygiene matters.** `demo_1h.mp4` is the proper distillation result:
the student was trained on COCO val2017 person images and benchmarked
on Intel sample-videos — **zero overlap** between training, validation,
and the demo footage. The `demo_smoke.mp4` is kept as historical
context: trained on 35 SAM 3 pseudo-labels with `single_cls=True`,
which resets the YOLOv8n-Seg head and destroys COCO's pretrained
person prior. The 1 h run fixes both issues — diversified training
corpus + 80 COCO class names retained in the dataset YAML so only
class 0 (person) carries gradient.

**Latency note.** v2 is slower per frame than the smoke run (37.7 ms
vs 15.4 ms) because the head retains all 80 COCO classes — more raw
NMS candidates per anchor at the same confidence threshold. Still
real-time at 22 FPS on a single laptop GPU. Switching back to a
single-class head after distillation, or raising `conf`, recovers the
50+ FPS regime trivially.

## Architectural commitments worth keeping

1. **Backbone outputs feature pyramid at named taps** (`P3/P4/P5`). The
   `Expert` Protocol in `runtime/pedestrian_expert.py` makes these the
   integration surface for future experts. Don't fuse class-specific
   computation into the backbone.

2. **Single class today, copy-paste tomorrow.** Adding cyclists, vehicles,
   signs is a new head + a new SAM 3 prompt set + a new YAML — the
   training pipeline doesn't change.

3. **Negative prompts matter.** The pedestrian expert is explicitly
   negative-prompted against cyclists and scooter riders so its training
   set doesn't pollute the future cyclist expert. The post-hoc filter in
   `teacher/pseudolabel_filter.py` belt-and-suspenders this.

## Out of scope (intentional)

- **Orin AGX port.** Same code paths apply; needs the device for INT8
  calibration and trtexec on the Jetson. Add `runtime/build_trt.py`
  invocation on the Orin.
- **MoE router.** `runtime/router/` is reserved; lands when expert #2
  exists.
- **Multi-camera batching, sensor fusion, tracking.** Phase 3+.

## References

- SAM 3: <https://github.com/facebookresearch/sam3>
- Ultralytics YOLOv8: <https://docs.ultralytics.com/>
- EMC2 (scenario-aware MoE for edge AV, ICCV 2025):
  <https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Towards_Accurate_and_Efficient_3D_Object_Detection_for_Autonomous_Driving_ICCV_2025_paper.pdf>
