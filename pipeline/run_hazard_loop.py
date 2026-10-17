"""Day 1 deliverable: standalone real-time hazard-alert loop. NO LLM.

capture -> detector -> tts, with cooldown de-duplication and full JSONL
logging. This is the isolated proactive layer; orchestrator.py is what merges
it with the reactive layer.

Run:  python -m pipeline.run_hazard_loop
      python -m pipeline.run_hazard_loop --headless --seconds 30

Acceptance test: wave an object close to the camera; you should hear a spoken
alert and see "alert" lines appear in logs/session_log.jsonl.
"""

from __future__ import annotations

import argparse
import time

import cv2

from pipeline.capture import ThreadedCamera
from pipeline.config_util import load_config, resolve_path
from pipeline.detector import Detector
from pipeline.logging_util import EventLogger
from pipeline.prompt_library import PromptLibrary
from pipeline.scene_state import Severity
from pipeline.tts import SpeechGate, TTSWorker


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone hazard-alert loop (day 1)")
    parser.add_argument("--headless", action="store_true", help="no preview window")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="auto-stop after N seconds (0 = run until 'q'/Ctrl-C)")
    parser.add_argument("--min-severity", default="medium",
                        choices=["low", "medium", "high", "critical"],
                        help="lowest severity that is spoken aloud")
    args = parser.parse_args()

    cfg = load_config()
    logger = EventLogger(resolve_path(cfg["logging"]["session_log"]))
    templates = PromptLibrary().by_category("hazard_alert").templates
    min_sev = Severity.parse(args.min_severity)

    detector = Detector(cfg, logger=logger)
    gate = SpeechGate(
        cooldown_s=cfg["arbitration"]["alert_cooldown_s"],
        min_gap_s=cfg["arbitration"]["min_gap_between_alerts_s"],
    )
    tts = TTSWorker(rate=cfg["tts"]["rate"], volume=cfg["tts"]["volume"]).start()
    camera = ThreadedCamera(
        device_index=cfg["camera"]["device_index"],
        width=cfg["camera"]["frame_width"],
        height=cfg["camera"]["frame_height"],
        fps=cfg["camera"]["fps"],
    )

    logger.log("system", component="run_hazard_loop", event="start",
               min_severity=min_sev.name)
    print(f"[hazard loop] speaking alerts at severity >= {min_sev.name}. "
          f"{'Ctrl-C' if args.headless else 'Press q'} to stop.")

    every_n = cfg["camera"].get("throttle_every_n_frames", 3)
    started = time.monotonic()
    frame_idx = 0

    camera.open()
    try:
        while True:
            if args.seconds and (time.monotonic() - started) >= args.seconds:
                break
            frame = camera.read()
            if frame is None:
                time.sleep(0.03)
                continue

            frame_idx += 1
            if frame_idx % every_n == 0:
                for obj in detector.detect(frame):
                    if obj.severity < min_sev:
                        continue
                    if not gate.allows(obj.key, bypass_gap=(obj.severity >= Severity.CRITICAL)):
                        continue
                    template = templates.get(obj.severity.name.lower(), "{label} {position_phrase}.")
                    phrase = template.format(
                        label=obj.label,
                        position_phrase=obj.spoken_position(),
                        distance_phrase=obj.spoken_distance(),
                    )
                    tts.say(phrase, priority=int(Severity.CRITICAL) - int(obj.severity), kind="alert")
                    gate.mark(obj.key)
                    print(f"[alert:{obj.severity.name}] {phrase}")
                    logger.log("alert", object_class=obj.label, position=obj.position,
                               distance_m=round(obj.distance_m, 2) if obj.distance_m else None,
                               severity=obj.severity.name, text=phrase)

            if not args.headless:
                cv2.imshow("hazard loop (q to quit)", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        logger.log("system", component="run_hazard_loop", event="stop")
        tts.stop()
        camera.release()
        if not args.headless:
            cv2.destroyAllWindows()
        logger.close()
        print("[hazard loop] stopped.")


if __name__ == "__main__":
    main()
