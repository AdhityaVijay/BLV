"""Why isn't it alerting? Prints the full decision chain for live frames.

    python -m pipeline.debug_alerts
    python -m pipeline.debug_alerts --image test_data/images/chair_01.jpg
    python -m pipeline.debug_alerts --seconds 20 --all-classes

An alert has to survive five gates, and silence looks identical at every one
of them. This prints each gate's verdict per object, so the failure is
attributable instead of guessed at:

    1. DETECTED       did YOLO see anything at all, above the conf threshold?
    2. RELEVANT       is the class in detector.relevant_classes?
    3. DISTANCE       what did the pinhole estimate give, and which band?
    4. SEVERITY       did it reach MEDIUM, the floor for being spoken?
    5. GATE           cooldown / minimum gap between utterances

Most "it never speaks" reports are gate 4, and most of those are a truncated
bounding box: sitting at a laptop, only your head and shoulders are in frame,
so the box is short, and a short box reads as FAR AWAY to a pinhole estimate
that assumes it can see the whole object.
"""

from __future__ import annotations

import argparse
import time

import cv2

from pipeline.arbitration import Arbiter, TurnState
from pipeline.capture import Camera
from pipeline.config_util import load_config, resolve_path
from pipeline.detector import Detector
from pipeline.scene_state import Severity
from pipeline.tts import SpeechGate


def describe_thresholds(cfg: dict) -> None:
    d = cfg["detector"]
    a = cfg["arbitration"]
    z = d["danger_zone"]
    print("Thresholds in effect (config.yaml):")
    print(f"  confidence            >= {d['confidence_threshold']}")
    print(f"  distance bands (m)     imminent <{d['distance_bands_m']['imminent']}  "
          f"near <{d['distance_bands_m']['near']}  mid <{d['distance_bands_m']['mid']}")
    print(f"  area fallback          >= {d['area_fraction_alert']} of frame")
    print(f"  walking path           x {z['x_min']}-{z['x_max']}, "
          f"object bottom below y {z['y_min']}")
    print(f"  alert cooldown         {a['alert_cooldown_s']}s same phrase, "
          f"{a['min_gap_between_alerts_s']}s between any two")
    print(f"  relevant classes       {len(d.get('relevant_classes') or [])} "
          f"(empty = all 80 COCO classes)")
    print("  SPOKEN FLOOR           severity >= MEDIUM\n")


def analyse(frame, detector: Detector, gate: SpeechGate, arbiter: Arbiter,
            all_classes: bool) -> None:
    h, w = frame.shape[:2]

    # Gate 1+2: what survived detection and class filtering.
    objects = detector.detect(frame, log_detections=False)

    if not objects:
        # Re-run without the class filter to tell "saw nothing" apart from
        # "saw something irrelevant" - a distinction that changes the fix.
        saved, detector.relevant = detector.relevant, set()
        unfiltered = detector.detect(frame, log_detections=False)
        detector.relevant = saved
        if unfiltered:
            names = ", ".join(sorted({o.label for o in unfiltered}))
            print(f"  DETECTED but ALL FILTERED OUT by relevant_classes: {names}")
            print("  -> add the class to detector.relevant_classes, or empty the list")
        else:
            print("  NOTHING DETECTED above the confidence threshold.")
            print("  -> is the lens covered? is the scene too dark? try lowering")
            print("     detector.confidence_threshold, or step fully into frame")
        return

    print(f"  {len(objects)} object(s) detected:")
    would_speak = 0
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        box_h = y2 - y1
        truncated = y1 <= 2 or y2 >= h - 2 or x1 <= 2 or x2 >= w - 2
        dist = f"{obj.distance_m:.2f}m" if obj.distance_m is not None else "unknown"

        speaks = obj.severity >= Severity.MEDIUM
        gate_ok = gate.allows(obj.key, bypass_gap=(obj.severity >= Severity.CRITICAL))
        decision = arbiter.decide(obj.severity, TurnState.IDLE, gate_ok)

        mark = "SPEAKS" if (speaks and decision.speaks_now) else "silent"
        print(f"    [{mark}] {obj.label:<12} conf {obj.confidence:.2f}  "
              f"box_h {box_h:>5.0f}px  dist {dist:<9} band {obj.distance_band:<8} "
              f"path {'Y' if obj.in_danger_zone else 'n'}  sev {obj.severity.name}")

        if not speaks:
            print(f"             -> below the MEDIUM floor, so never spoken")
            if obj.distance_m is not None and obj.distance_m > 2.0:
                print(f"             -> estimated {obj.distance_m:.1f}m away; "
                      f"get closer or raise distance_bands_m.near")
            if truncated:
                print("             -> BOX TOUCHES THE FRAME EDGE: the object is cut")
                print("                off, so its box is shorter than the real object")
                print("                and the distance estimate is too LARGE. Step back")
                print("                so you fit entirely in view.")
            if not obj.in_danger_zone:
                print("             -> outside the walking path, so no escalation")
        elif not decision.speaks_now:
            print(f"             -> generated but {decision.decision.value}: {decision.reason}")
        else:
            would_speak += 1
            gate.mark(obj.key)
            print(f"             -> \"{obj.alert_phrase()}\"")

    if would_speak == 0:
        print("  NO OBJECT REACHED THE SPOKEN FLOOR THIS FRAME.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why alerts aren't firing")
    parser.add_argument("--image", type=str, help="analyse a single image instead of the camera")
    parser.add_argument("--seconds", type=float, default=30.0, help="how long to watch")
    parser.add_argument("--every", type=float, default=1.5, help="seconds between reports")
    parser.add_argument("--all-classes", action="store_true",
                        help="ignore relevant_classes and report every COCO class")
    args = parser.parse_args()

    cfg = load_config()
    if args.all_classes:
        cfg["detector"]["relevant_classes"] = []

    describe_thresholds(cfg)

    detector = Detector(cfg, logger=None)
    gate = SpeechGate(
        cooldown_s=cfg["arbitration"]["alert_cooldown_s"],
        min_gap_s=cfg["arbitration"]["min_gap_between_alerts_s"],
    )
    arbiter = Arbiter(
        interrupt_threshold=Severity.parse(cfg["arbitration"]["interrupt_threshold"]),
        queue_threshold=Severity.parse(cfg["arbitration"]["queue_threshold"]),
    )

    if args.image:
        path = resolve_path(args.image)
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Could not read {path}")
        print(f"--- {path.name} ---")
        analyse(frame, detector, gate, arbiter, args.all_classes)
        return

    cam = Camera(
        device_index=cfg["camera"]["device_index"],
        width=cfg["camera"]["frame_width"],
        height=cfg["camera"]["frame_height"],
        fps=cfg["camera"]["fps"],
    )
    cam.open()
    print(f"Watching for {args.seconds:.0f}s. Move around, step back, walk toward "
          f"the camera.\nCtrl-C to stop early.\n")

    started = time.monotonic()
    try:
        while (elapsed := time.monotonic() - started) < args.seconds:
            frame = cam.read()
            if frame is None:
                time.sleep(0.05)
                continue
            print(f"--- t={elapsed:5.1f}s ---")
            analyse(frame, detector, gate, arbiter, args.all_classes)
            print()
            time.sleep(args.every)
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
    print("Done. If everything read 'silent', the numbers above say which")
    print("threshold to move in config.yaml.")


if __name__ == "__main__":
    main()
