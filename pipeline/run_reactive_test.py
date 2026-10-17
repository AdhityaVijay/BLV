"""Day 2 deliverable: isolated reactive-layer test + latency measurement.

Two modes:

  --webcam         capture one frame, send it to the VLM, speak the answer
  --images         run over every image in test_data/images/ and print a
                   latency table, so the numbers are measured on your actual
                   RTX 4060 rather than assumed

Run:  python -m pipeline.run_reactive_test --images
      python -m pipeline.run_reactive_test --webcam --speak
      python -m pipeline.run_reactive_test --images --repeat 3
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2

from pipeline.capture import Camera
from pipeline.config_util import load_config, resolve_path
from pipeline.logging_util import EventLogger
from pipeline.scene_state import SceneState
from pipeline.tts import speak
from pipeline.vlm import VLM

DEFAULT_PROMPT = ("Describe what's in front of me in one or two sentences, "
                  "in plain language, without visual jargon.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reactive VLM layer test (day 2)")
    parser.add_argument("--webcam", action="store_true", help="use one live webcam frame")
    parser.add_argument("--images", action="store_true",
                        help="run over test_data/images/ and report latency")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--speak", action="store_true", help="read responses aloud")
    parser.add_argument("--repeat", type=int, default=1,
                        help="repeat each image N times (first call includes model load)")
    parser.add_argument("--no-grounding", action="store_true",
                        help="skip detector grounding (day-2 default: no detector running anyway)")
    args = parser.parse_args()

    if not args.webcam and not args.images:
        args.images = True

    cfg = load_config()
    logger = EventLogger(resolve_path(cfg["logging"]["session_log"]))
    # Day 2 keeps the layers isolated: no detector, so no grounding facts exist.
    scene = SceneState(grounding_enabled=not args.no_grounding)
    vlm = VLM(cfg, scene_state=scene, logger=logger)

    avail = vlm.check_available()
    if not avail.get("ok"):
        print(f"[vlm] Ollama unavailable: {avail.get('note') or avail.get('error')}")
        print(f"      Start it with 'ollama serve', then: ollama pull {cfg['vlm']['model']}")
        return
    print(f"[vlm] model={avail['active_model']} host={avail['host']}")

    latencies: list[float] = []

    if args.webcam:
        with Camera(cfg["camera"]["device_index"], cfg["camera"]["frame_width"],
                    cfg["camera"]["frame_height"], cfg["camera"]["fps"]) as cam:
            for _ in range(5):      # discard the first frames: webcams need to settle
                frame = cam.read()
        if frame is None:
            print("[error] no frame from webcam")
            return
        resp = vlm.ask(frame, args.prompt, category="scene_description", continue_session=False)
        latencies.append(resp.latency_ms)
        print(f"\n[webcam] {resp.latency_ms:7.0f} ms  {resp.text}")
        if args.speak and resp.text:
            speak(resp.text, rate=cfg["tts"]["rate"])

    if args.images:
        img_dir = resolve_path("test_data/images")
        paths = sorted(
            p for p in img_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not paths:
            print(f"\n[images] no images found in {img_dir}")
            print("         Drop 5+ photos there (a cluttered desk, a corridor, a")
            print("         low-light room, a partially hidden object) and re-run.")
        else:
            print(f"\n[images] {len(paths)} image(s), {args.repeat} pass(es) each\n")
            print(f"{'image':<34}{'pass':>5}{'latency_ms':>12}  response")
            print("-" * 100)
            for path in paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    print(f"{path.name:<34}  [unreadable]")
                    continue
                for i in range(args.repeat):
                    resp = vlm.ask(frame, args.prompt, category="scene_description",
                                   continue_session=False)
                    latencies.append(resp.latency_ms)
                    text = (resp.text or resp.error or "")[:56].replace("\n", " ")
                    print(f"{path.name:<34}{i + 1:>5}{resp.latency_ms:>12.0f}  {text}")
                    if args.speak and resp.text and i == 0:
                        speak(resp.text, rate=cfg["tts"]["rate"])

    if latencies:
        print("\n" + "-" * 100)
        print(f"calls={len(latencies)}  "
              f"median={statistics.median(latencies):.0f} ms  "
              f"min={min(latencies):.0f} ms  max={max(latencies):.0f} ms")
        if len(latencies) > 1:
            print(f"NOTE: the first call includes model load into VRAM; the median "
                  f"excluding it is {statistics.median(latencies[1:]):.0f} ms.")
        print("These are your real numbers — use them in the write-up, not estimates.")
    logger.close()


if __name__ == "__main__":
    main()
