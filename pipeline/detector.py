"""YOLOv8n wrapper + proactive hazard-alert rule.

DISTANCE ESTIMATION — read this before tuning anything.

The obvious heuristic ("bbox taller than X% of the frame means close") is
wrong in a way that matters here: it conflates object SIZE with object
DISTANCE. A bus twenty metres away produces a taller box than a chair at arm's
length, so the naive rule calls the bus "very close" and stays silent about the
chair you are about to walk into. For a system whose whole purpose is not
walking into things, that is a safety-relevant defect, not an imprecision.

Instead we use the pinhole camera model with per-class known heights:

        distance  =  (real_object_height_m  x  focal_length_px) / bbox_height_px
        focal_length_px  =  (frame_width_px / 2) / tan(horizontal_fov / 2)

This is still an estimate. Its assumptions, stated plainly:
  * the object is roughly its typical real-world height (a child is not 1.7 m);
  * the object is fully visible in frame (a truncated box reads as far away);
  * the camera is not heavily rotated relative to the object.
Classes with no known height fall back to the frame-area rule and report
distance_m = None rather than inventing a number.

Accuracy in practice is roughly +/-20-30% at 1-4 m, which is enough to
separate "less than a metre" from "three metres" — the granularity speech
actually needs. Measuring your camera's true FOV (config.yaml) tightens it.
"""

from __future__ import annotations

import math
import time

import numpy as np

from pipeline.scene_state import Severity, TrackedObject

# Typical real-world heights in metres, for the pinhole distance estimate.
# Sources are ordinary furniture/vehicle dimensions; adjust for your setting.
KNOWN_HEIGHTS_M: dict[str, float] = {
    "person": 1.70,
    "bicycle": 1.10,
    "car": 1.50,
    "motorcycle": 1.10,
    "bus": 3.00,
    "truck": 3.20,
    "dog": 0.55,
    "cat": 0.30,
    "chair": 0.90,
    "couch": 0.80,
    "bench": 0.90,
    "dining table": 0.75,
    "potted plant": 0.60,
    "refrigerator": 1.70,
    "tv": 0.60,
    "laptop": 0.25,
    "backpack": 0.45,
    "suitcase": 0.60,
    "bottle": 0.25,
    "cup": 0.10,
    "traffic light": 0.80,
    "stop sign": 0.75,
    "fire hydrant": 0.75,
    "bed": 0.60,
    "toilet": 0.75,
    "microwave": 0.30,
    "oven": 0.85,
    "sink": 0.20,
}


def focal_length_px(frame_width: int, horizontal_fov_deg: float) -> float:
    """Derive focal length in pixels from the camera's horizontal FOV."""
    half_fov_rad = math.radians(horizontal_fov_deg) / 2.0
    return (frame_width / 2.0) / math.tan(half_fov_rad)


def estimate_distance_m(
    label: str, bbox_height_px: float, focal_px: float
) -> float | None:
    """Pinhole distance estimate. Returns None for unknown classes."""
    real_h = KNOWN_HEIGHTS_M.get(label)
    if real_h is None or bbox_height_px <= 1.0:
        return None
    return (real_h * focal_px) / bbox_height_px


def distance_band(distance_m: float | None, area_fraction: float, bands: dict, area_alert: float) -> str:
    """Bucket a distance for phrasing and de-duplication. Falls back to the
    frame-area rule when no metric estimate is available."""
    if distance_m is None:
        return "imminent" if area_fraction >= area_alert else "unknown"
    if distance_m < bands.get("imminent", 1.0):
        return "imminent"
    if distance_m < bands.get("near", 2.0):
        return "near"
    if distance_m < bands.get("mid", 4.0):
        return "mid"
    return "far"


def horizontal_position(center_x: float, frame_width: int) -> str:
    """Left/centre/right thirds. Matches the spatial convention used in the
    prompt library, so proactive and reactive speech agree."""
    r = center_x / max(frame_width, 1)
    if r < 0.35:
        return "left"
    if r > 0.65:
        return "right"
    return "centre"


def classify_severity(
    *,
    band: str,
    in_danger_zone: bool,
    area_fraction: float,
    label: str,
    high_risk_classes: set[str],
    area_alert: float,
) -> Severity:
    """The proactive-alert rule. Kept dead simple and explicit (day-1
    constraint) so "how close is close" is one readable function to tune.

        CRITICAL - imminent (<1 m) AND in the walking path      -> interrupts speech
        HIGH     - imminent anywhere, or near (<2 m) in the path, or a very
                   large box (fallback for unknown-height classes)
        MEDIUM   - near anywhere, or mid-range in the path
        LOW      - everything else

    High-risk classes (people, vehicles, bicycles, dogs) escalate one level
    when they are in the walking path: they move toward you and a collision
    costs more.
    """
    if band == "imminent" and in_danger_zone:
        sev = Severity.CRITICAL
    elif band == "imminent" or (band == "near" and in_danger_zone) or area_fraction >= area_alert:
        sev = Severity.HIGH
    elif band == "near" or (band == "mid" and in_danger_zone):
        sev = Severity.MEDIUM
    else:
        sev = Severity.LOW

    if label in high_risk_classes and in_danger_zone and sev < Severity.CRITICAL:
        sev = sev.bump(1)
    return sev


class Detector:
    def __init__(self, cfg: dict, logger=None, verbose_device: bool = True):
        from ultralytics import YOLO  # imported here so unit tests don't need it

        d = cfg["detector"]
        c = cfg["camera"]
        self.cfg = cfg
        self.logger = logger

        self.device = self._resolve_device(d.get("device", "auto"))
        self.model = YOLO(d["model_path"])
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
            self.model.to("cpu")

        self.conf = d["confidence_threshold"]
        self.relevant = set(d.get("relevant_classes") or [])
        self.high_risk = set(d.get("high_risk_classes") or [])
        self.area_alert = d["area_fraction_alert"]
        self.bands = d["distance_bands_m"]
        self.zone = d["danger_zone"]
        self.focal_px = focal_length_px(c["frame_width"], c["horizontal_fov_deg"])

        if verbose_device:
            self._report_device()

    # -- device ------------------------------------------------------------

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _report_device(self) -> None:
        """Day-1 constraint: confirm on startup whether YOLO is on GPU or CPU.
        Silent CPU fallback is the single most common cause of 'why is this so
        slow', so it is printed loudly rather than logged quietly."""
        info = {"device": self.device}
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
        except Exception as e:
            info["torch_error"] = str(e)

        banner = f"[detector] YOLOv8 running on: {self.device.upper()}"
        if info.get("gpu_name"):
            banner += f" ({info['gpu_name']}, {info.get('vram_gb')} GB)"
        print(banner)
        if self.device == "cpu":
            print("[detector] WARNING: running on CPU. Expect much lower frame rates. "
                  "Install a CUDA build of torch to use the GPU.")
        if self.logger:
            self.logger.log("system", component="detector", **info)

    # -- detection ---------------------------------------------------------

    def _in_danger_zone(self, x1: float, x2: float, y2: float, w: int, h: int) -> bool:
        """True when the object's horizontal centre sits in the central
        corridor AND its BOTTOM edge is in the lower part of the frame (i.e.
        it is standing on the floor ahead of you, not hanging high in view)."""
        cx = (x1 + x2) / 2.0
        return (
            self.zone["x_min"] * w <= cx <= self.zone["x_max"] * w
            and y2 >= self.zone["y_min"] * h
        )

    def detect(self, frame: np.ndarray, log_detections: bool = True) -> list[TrackedObject]:
        t0 = time.perf_counter()
        results = self.model.predict(frame, conf=self.conf, verbose=False, device=self.device)
        infer_ms = (time.perf_counter() - t0) * 1000.0

        h, w = frame.shape[:2]
        frame_area = float(w * h)
        objects: list[TrackedObject] = []

        for result in results:
            names = result.names
            for box in result.boxes:
                label = names[int(box.cls[0])]
                if self.relevant and label not in self.relevant:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                bbox_h = y2 - y1
                area_fraction = ((x2 - x1) * bbox_h) / frame_area
                dist = estimate_distance_m(label, bbox_h, self.focal_px)
                band = distance_band(dist, area_fraction, self.bands, self.area_alert)
                in_zone = self._in_danger_zone(x1, x2, y2, w, h)
                sev = classify_severity(
                    band=band,
                    in_danger_zone=in_zone,
                    area_fraction=area_fraction,
                    label=label,
                    high_risk_classes=self.high_risk,
                    area_alert=self.area_alert,
                )
                obj = TrackedObject(
                    label=label,
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                    position=horizontal_position((x1 + x2) / 2.0, w),
                    distance_m=dist,
                    distance_band=band,
                    area_fraction=area_fraction,
                    in_danger_zone=in_zone,
                    severity=sev,
                )
                objects.append(obj)

                if log_detections and self.logger:
                    self.logger.log(
                        "detection",
                        object_class=label,
                        position=obj.position,
                        distance_m=round(dist, 2) if dist is not None else None,
                        distance_band=band,
                        area_fraction=round(area_fraction, 4),
                        in_danger_zone=in_zone,
                        severity=sev.name,
                        confidence=round(obj.confidence, 3),
                        frame_latency_ms=round(infer_ms, 1),
                    )

        # Nearest and most severe first — so the orchestrator speaks the thing
        # most likely to hurt you first, rather than whatever YOLO listed first.
        objects.sort(key=lambda o: (-int(o.severity), o.distance_m if o.distance_m is not None else 1e6))
        return objects
