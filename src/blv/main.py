"""Entry point: opens the webcam, runs throttled YOLOv8 obstacle
detection every frame, speaks cooldown-managed alerts, and runs a BLIP
scene description on demand when the user presses the trigger key.

Run with:  python -m blv.main
(from the src/ directory, with the venv active — see README.md)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import yaml

from blv.audio.alert_manager import AlertManager
from blv.audio.tts_engine import TTSEngine
from blv.capture.camera import Camera
from blv.description.scene_describer import SceneDescriber
from blv.detection.object_detector import ObjectDetector
from blv.utils.geometry import describe_position
from blv.utils.logger import get_logger

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    log = get_logger(__name__, cfg["logging"]["level"], cfg["logging"].get("log_file"))

    cam_cfg = cfg["camera"]
    det_cfg = cfg["detection"]
    desc_cfg = cfg["description"]
    audio_cfg = cfg["audio"]

    log.info("Loading object detector...")
    detector = ObjectDetector(
        model_path=det_cfg["model_path"],
        confidence_threshold=det_cfg["confidence_threshold"],
        obstacle_classes=det_cfg.get("obstacle_classes") or None,
    )

    log.info("Loading scene describer (this can take a while on first run)...")
    describer = SceneDescriber(model_name=desc_cfg["model_name"], device=desc_cfg["device"])

    tts = TTSEngine(rate=audio_cfg["rate"], volume=audio_cfg["volume"])
    alerts = AlertManager(
        alert_cooldown_seconds=audio_cfg["alert_cooldown_seconds"],
        min_priority_gap_seconds=audio_cfg["min_priority_gap_seconds"],
    )

    trigger_key = ord(desc_cfg["trigger_key"])
    run_every_n = max(1, det_cfg["run_every_n_frames"])
    frame_idx = 0

    log.info("Starting camera. Press 'q' to quit, '%s' to describe the scene.", desc_cfg["trigger_key"])
    with Camera(cam_cfg["device_index"], cam_cfg["frame_width"], cam_cfg["frame_height"], cam_cfg["fps"]) as cam:
        while True:
            frame = cam.read()
            if frame is None:
                log.warning("Empty frame from camera, stopping.")
                break

            frame_idx += 1
            if frame_idx % run_every_n == 0:
                detections = detector.detect(frame)
                h, w = frame.shape[:2]
                for det in detections:
                    position = describe_position(det.box, w, h)
                    key = f"{det.label}:{position}"
                    if alerts.should_speak(key):
                        phrase = f"{det.label} {position}"
                        log.info("ALERT: %s", phrase)
                        tts.speak(phrase)
                        alerts.mark_spoken(key)

            cv2.imshow("BLV (press q to quit, d to describe)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == trigger_key:
                log.info("Describing scene...")
                caption = describer.describe(frame, max_new_tokens=desc_cfg["max_new_tokens"])
                log.info("SCENE: %s", caption)
                tts.speak(caption)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
