"""Thin wrapper around cv2.VideoCapture so the rest of the codebase
doesn't touch OpenCV's capture API directly (makes it easy to swap in
a video file or a phone IP-camera stream later).
"""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.device_index}. "
                "Check config.yaml -> camera.device_index."
            )

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Camera not opened — call open() first.")
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
