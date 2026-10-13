"""Real-time obstacle/object detection using a pretrained YOLOv8 model
(via the `ultralytics` package). Runs on CPU fine at a throttled frame
rate — see config.yaml -> detection.run_every_n_frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from blv.utils.geometry import BoundingBox


@dataclass
class Detection:
    label: str
    confidence: float
    box: BoundingBox


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        confidence_threshold: float = 0.45,
        obstacle_classes: list[str] | None = None,
    ):
        # ultralytics auto-downloads yolov8n.pt the first time this runs
        # if the file isn't already present at model_path.
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.obstacle_classes = set(obstacle_classes) if obstacle_classes else None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections: list[Detection] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls[0])]
                if self.obstacle_classes and label not in self.obstacle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=label,
                        confidence=float(box.conf[0]),
                        box=BoundingBox(x1, y1, x2, y2),
                    )
                )
        return detections
