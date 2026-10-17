"""Webcam capture + frame sampling.

Two capture modes, for two different consumers:

  Camera         - simple, synchronous. Exposes frames(every_n=...) as a
                   generator. Used by the standalone day-1/day-2 scripts.

  ThreadedCamera - background reader thread that keeps only the LATEST frame.
                   Used by the orchestrator. This matters more than it looks:
                   OpenCV buffers frames internally, so if the main loop stalls
                   (e.g. while speaking), a subsequent read() returns a STALE
                   buffered frame. An obstacle system reasoning about a frame
                   from two seconds ago is worse than useless, so the reader
                   thread continuously drains the buffer and we always analyse
                   the most recent frame available.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Iterator

import cv2
import numpy as np


def default_backend() -> int:
    """Pick the capture backend.

    On Windows, OpenCV defaults to Microsoft Media Foundation (MSMF), which
    routinely takes 10-30 seconds to open a webcam and sometimes blocks
    indefinitely with no error. DirectShow opens immediately. This is the
    single most common reason the pipeline appears to hang at startup on
    Windows, so it is the default there rather than an option to discover.
    """
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


class Camera:
    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backend: int | None = None,
    ):
        self.backend = default_backend() if backend is None else backend
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        t0 = time.monotonic()
        print(f"[camera] opening device {self.device_index} "
              f"(backend={'DSHOW' if self.backend == cv2.CAP_DSHOW else self.backend})...")
        self._cap = cv2.VideoCapture(self.device_index, self.backend)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Ask the driver for a 1-frame buffer where supported; reduces the
        # staleness problem at the source. Not honoured by every backend.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.device_index} "
                f"after {time.monotonic() - t0:.1f}s.\n"
                "  - Try a different camera.device_index in config.yaml (0, 1, 2).\n"
                "  - Windows: Settings > Privacy & security > Camera > "
                "'Let desktop apps access your camera' must be ON.\n"
                "  - Close anything else using the webcam (Teams, Zoom, the Camera app)."
            )
        print(f"[camera] opened in {time.monotonic() - t0:.1f}s")
        return self

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            raise RuntimeError("Camera not opened - call open() first.")
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self, every_n: int = 1) -> Iterator[np.ndarray]:
        """Yield frames, optionally throttled to every Nth frame.

        every_n=1 yields every frame (default); every_n=3 yields every third.
        Skipped frames are still read from the device so the buffer drains.
        """
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        idx = 0
        while True:
            frame = self.read()
            if frame is None:
                return
            idx += 1
            if idx % every_n == 0:
                yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class ThreadedCamera(Camera):
    """Camera whose reader runs in its own thread, always exposing the newest
    frame. read() never blocks on the device and never returns a stale frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latest: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> "ThreadedCamera":
        super().open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader, name="capture", daemon=True)
        self._thread.start()
        # Wait for the first frame so callers don't get None immediately.
        #
        # If none arrives we RAISE rather than returning quietly. Previously
        # this returned self after the timeout, which meant a camera that
        # opened but delivered no frames produced no window, no error and no
        # output at all - the process just sat there looking alive. A loud
        # failure here is worth more than a graceful one.
        timeout = 8.0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest is not None:
                    return self
            time.sleep(0.02)

        self.release()
        raise RuntimeError(
            f"Camera {self.device_index} opened but delivered no frames in {timeout:.0f}s.\n"
            "  - Another application may be holding the camera (Teams, Zoom, Camera app).\n"
            "  - Windows: Settings > Privacy & security > Camera must allow desktop apps.\n"
            "  - Try another camera.device_index in config.yaml (0, 1, 2).\n"
            "  - Laptop privacy shutter closed, or a hardware camera-off key toggled?"
        )

    def _reader(self) -> None:
        while not self._stop.is_set():
            frame = super().read()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
                self._latest_ts = time.monotonic()

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def read_with_age(self) -> tuple[np.ndarray | None, float]:
        """Return (frame, seconds_since_captured) — lets callers detect a
        frozen camera rather than silently analysing an old frame."""
        with self._lock:
            if self._latest is None:
                return None, float("inf")
            return self._latest.copy(), time.monotonic() - self._latest_ts

    def frames(self, every_n: int = 1) -> Iterator[np.ndarray]:
        """Yield the newest frame at the device's pace / every_n.

        Unlike the synchronous version this cannot fall behind, because the
        reader thread has already discarded intermediate frames.
        """
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        interval = every_n / float(self.fps or 30)
        while not self._stop.is_set():
            frame = self.read()
            if frame is not None:
                yield frame
            time.sleep(interval)

    def release(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().release()
