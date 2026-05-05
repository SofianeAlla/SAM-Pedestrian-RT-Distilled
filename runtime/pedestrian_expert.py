"""Pedestrian expert inference wrapper.

Single-class detector + segmenter. Loads an Ultralytics YOLO checkpoint
(.pt) or an ONNX model and exposes a stable inference API that downstream
code (and future MoE experts) can depend on.

This file also defines the Expert plugin contract that future experts
(vehicle, cyclist, sign, ...) will implement, so the MoE foundation can
be built without rewriting this prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Detection:
    """Single detection result."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None
    class_id: int = 0
    class_name: str = "pedestrian"


@dataclass
class InferenceResult:
    """Container returned by Expert.infer."""

    detections: list[Detection] = field(default_factory=list)
    image_hw: tuple[int, int] = (0, 0)
    latency_ms: float = 0.0


@runtime_checkable
class Expert(Protocol):
    """Contract every MoE expert must implement.

    Lets the future router treat all experts uniformly. The pedestrian
    expert below is the reference implementation.
    """

    name: str
    class_names: tuple[str, ...]

    def infer(self, image_bgr: np.ndarray) -> InferenceResult: ...

    @property
    def backbone_taps(self) -> dict[str, Any]:
        """Names of feature-pyramid taps the router can read.

        Future MoE: the shared backbone runs once per frame; the router
        reads these taps to decide which experts to fire; each expert's
        head consumes them.
        """
        ...


class PedestrianExpert:
    """SAM 3-distilled YOLOv8n-Seg pedestrian expert.

    Wraps an Ultralytics YOLO model trained on SAM 3 pseudo-labels.
    Single class, det + seg.
    """

    name = "pedestrian"
    class_names = ("pedestrian",)

    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda:0",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        half: bool = True,
    ) -> None:
        from ultralytics import YOLO

        self.weights = str(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.half = half
        self._model = YOLO(self.weights)
        # Warmup: a single dummy forward to populate caches.
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self._model.predict(
            dummy, imgsz=imgsz, device=device, half=half, verbose=False
        )

    def infer(self, image_bgr: np.ndarray) -> InferenceResult:
        import time

        t0 = time.perf_counter()
        results = self._model.predict(
            image_bgr,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            return InferenceResult(
                detections=[], image_hw=image_bgr.shape[:2], latency_ms=latency_ms
            )

        r = results[0]
        detections: list[Detection] = []
        boxes = r.boxes
        masks = r.masks

        if boxes is None or len(boxes) == 0:
            return InferenceResult(
                detections=[], image_hw=image_bgr.shape[:2], latency_ms=latency_ms
            )

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        if masks is not None:
            mask_arr = masks.data.cpu().numpy()
        else:
            mask_arr = None

        # If the model is a single-class model (typical for our distilled
        # student via single_cls=True), every detection is pedestrian by
        # definition — Ultralytics labels it "item" internally. Otherwise
        # filter to person/pedestrian classes.
        is_single_class = len(self._model.names) == 1

        for i in range(len(xyxy)):
            cid = int(classes[i])
            cname = self._model.names.get(cid, "").lower()
            if not is_single_class and cname not in {"pedestrian", "person"}:
                continue
            m = None
            if mask_arr is not None and i < len(mask_arr):
                m = mask_arr[i].astype(np.float32)
            detections.append(
                Detection(
                    box_xyxy=tuple(map(float, xyxy[i])),
                    score=float(confs[i]),
                    mask=m,
                    class_id=cid,
                    class_name=(
                        "pedestrian" if is_single_class else self._model.names.get(cid, "pedestrian")
                    ),
                )
            )

        return InferenceResult(
            detections=detections,
            image_hw=image_bgr.shape[:2],
            latency_ms=latency_ms,
        )

    @property
    def backbone_taps(self) -> dict[str, Any]:
        """Expose YOLO backbone feature-map names.

        For YOLOv8 the standard backbone taps are P3/P4/P5 (strides 8/16/32).
        Future MoE experts and the router consume these.
        """
        return {
            "P3": {"stride": 8, "module": "model.4"},
            "P4": {"stride": 16, "module": "model.6"},
            "P5": {"stride": 32, "module": "model.9"},
        }


# Sanity check at import time: PedestrianExpert satisfies the Expert protocol.
def _check_protocol() -> None:
    _: type[Expert] = PedestrianExpert  # type: ignore[assignment]


_check_protocol()
