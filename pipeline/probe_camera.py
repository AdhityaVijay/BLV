"""Find a working camera index and backend. Run this when the pipeline hangs
or reports no frames.

    python -m pipeline.probe_camera

Tries indices 0-3 against each available backend, reports which combinations
open AND actually deliver a frame (those are different things: a camera can
open and then hand you nothing), and prints the config.yaml line to use.
"""

from __future__ import annotations

import sys
import time

import cv2


def backends() -> list[tuple[str, int]]:
    if sys.platform == "win32":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    if sys.platform == "darwin":
        return [("AVFOUNDATION", cv2.CAP_AVFOUNDATION), ("ANY", cv2.CAP_ANY)]
    return [("V4L2", cv2.CAP_V4L2), ("ANY", cv2.CAP_ANY)]


def probe(index: int, backend_id: int, name: str) -> tuple[bool, str]:
    t0 = time.monotonic()
    cap = None
    try:
        cap = cv2.VideoCapture(index, backend_id)
        if not cap.isOpened():
            return False, f"did not open ({time.monotonic() - t0:.1f}s)"
        ok, frame = cap.read()
        elapsed = time.monotonic() - t0
        if not ok or frame is None:
            return False, f"opened but no frame ({elapsed:.1f}s)"
        h, w = frame.shape[:2]
        return True, f"OK {w}x{h} in {elapsed:.1f}s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        if cap is not None:
            cap.release()


def main() -> None:
    print("Probing camera indices 0-3 across backends. MSMF can take 10-30s "
          "per attempt on Windows - that slowness is the point of this script.\n")
    working: list[tuple[int, str]] = []
    for index in range(4):
        for name, backend_id in backends():
            ok, detail = probe(index, backend_id, name)
            flag = "  OK  " if ok else "  --  "
            print(f"{flag} index={index}  backend={name:<12} {detail}")
            if ok:
                working.append((index, name))
        print()

    if not working:
        print("No working camera found. Check, in order:")
        print("  1. Windows: Settings > Privacy & security > Camera >")
        print("     'Let desktop apps access your camera' is ON")
        print("  2. Nothing else is holding the camera (Teams, Zoom, Camera app)")
        print("  3. Laptop privacy shutter / camera-off function key")
        print("  4. The camera works at all - open the Windows Camera app")
        return

    index, name = working[0]
    print(f"Use this in config.yaml:\n\n  camera:\n    device_index: {index}\n")
    if name != ("DSHOW" if sys.platform == "win32" else "V4L2"):
        print(f"Note: index {index} worked on the {name} backend. The pipeline "
              f"defaults to DSHOW on Windows; if it still fails, pass "
              f"backend=cv2.CAP_{name} in capture.py's default_backend().")


if __name__ == "__main__":
    main()
