"""GROUNDED SCENE-STATE BLACKBOARD — the novel contribution of this prototype.

--------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------
The standard hybrid design (and the one the day-3 spec describes) runs two
independent layers: a fast detector for proactive alerts, and a vision-LLM for
reactive conversation. They share a camera and an audio channel, and nothing
else. The orchestrator's only job is to stop them talking over each other.

That independence is the source of two known failure modes:

  1. SYCOPHANCY / HALLUCINATION. The VLM answers purely from pixels. When a
     user asserts something false ("there's nothing in front of me, right?"),
     the model has no external evidence to contradict them, and vision-language
     models are documented to fold to the user's framing. Meanwhile the
     detector, running on the same frame, may hold a high-confidence "person,
     1.2 m, dead ahead" — but nothing routes that fact to the layer doing the
     talking.

  2. INTENT-BLIND ALERTING. The hazard layer announces by fixed geometric
     rules. It has no idea the user just asked "where are my keys?" or "can I
     walk forward?", so it cannot raise the salience of the thing they care
     about, nor damp chatter while they are listening to an answer.

This module makes the two layers share state BIDIRECTIONALLY:

    detector  --[ tracked objects + metric geometry ]-->  blackboard --> VLM
    VLM       --[ parsed user intent                 ]-->  blackboard --> detector

The forward path turns the detector into a grounding oracle: its measurements
are injected into every VLM prompt as facts the model is instructed to trust
over the user's assertions. The backward path turns the conversation into a
salience signal: what the user said they are doing re-weights which hazards are
worth spending the audio channel on.

--------------------------------------------------------------------------
Why it is evaluable, not just architecture
--------------------------------------------------------------------------
`grounding.enabled` in config.yaml switches the forward path off. Day 6's
red-team harness runs the identical probe set with it on and off, so the
sycophancy/hallucination rate difference is a measured ablation rather than a
claim. That paired result is the evidence for the write-up.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


# ==========================================================================
# Severity
# ==========================================================================

class Severity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def parse(cls, name: str) -> "Severity":
        return cls[str(name).strip().upper()]

    def bump(self, levels: int) -> "Severity":
        """Shift severity, clamped to the enum range."""
        return Severity(max(int(Severity.LOW), min(int(Severity.CRITICAL), int(self) + levels)))


# ==========================================================================
# What the detector writes onto the blackboard
# ==========================================================================

@dataclass
class TrackedObject:
    """One detected object, with everything needed by BOTH consumers:
    the alert phrasing (position, distance, severity) and the VLM grounding
    block (label, confidence, metric distance)."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels
    position: str                             # "left" | "centre" | "right"
    distance_m: float | None                  # None when the class has no known height
    distance_band: str                        # imminent | near | mid | far | unknown
    area_fraction: float
    in_danger_zone: bool
    severity: Severity
    last_seen: float = field(default_factory=time.monotonic)

    @property
    def key(self) -> str:
        """Identity used for alert de-duplication: same object, same place,
        same proximity band should not be re-announced."""
        return f"{self.label}:{self.position}:{self.distance_band}"

    def spoken_position(self) -> str:
        return {"left": "to your left", "right": "to your right"}.get(self.position, "directly ahead")

    def spoken_distance(self) -> str:
        if self.distance_m is None:
            return {"imminent": "very close", "near": "close", "mid": "a few steps away"}.get(
                self.distance_band, "some distance away"
            )
        if self.distance_m < 1.0:
            return "less than a metre away"
        if self.distance_m < 1.6:
            return "about a metre away"
        if self.distance_m < 2.6:
            return "about two metres away"
        if self.distance_m < 4.0:
            return "about three metres away"
        return "several metres away"

    def alert_phrase(self) -> str:
        """Spatial convention: LEFT/RIGHT + DISTANCE, used identically by the
        proactive layer and (via the prompt library) the reactive layer."""
        return f"{self.label} {self.spoken_position()}, {self.spoken_distance()}"

    def grounding_line(self) -> str:
        """One fact line for the VLM prompt. Deliberately terse and numeric —
        this is evidence, not prose."""
        dist = f"{self.distance_m:.1f} m" if self.distance_m is not None else "distance unknown"
        zone = ", in the walking path" if self.in_danger_zone else ""
        return (
            f"- {self.label}: {self.position} of frame, {dist}{zone} "
            f"(detector confidence {self.confidence:.2f})"
        )


# ==========================================================================
# What the VLM writes back onto the blackboard
# ==========================================================================

class IntentMode:
    NAVIGATING = "navigating"   # user is moving / asked about walking
    SEARCHING = "searching"     # user is looking for a specific object
    INSPECTING = "inspecting"   # user is attending to a description
    UNKNOWN = "unknown"


# Keyword sets are intentionally small and readable rather than a learned
# classifier: this is a POC, and a wrong intent must be cheap to diagnose.
_NAV_WORDS = {
    "walk", "walking", "go", "going", "move", "moving", "ahead", "forward",
    "path", "doorway", "door", "exit", "safe", "cross", "step", "steps", "clear",
}
_SEARCH_WORDS = {"find", "where", "locate", "looking for", "my ", "which way to"}


@dataclass
class IntentProfile:
    """Parsed user intent — the backward path's payload."""

    raw_text: str = ""
    mode: str = IntentMode.UNKNOWN
    target_labels: tuple[str, ...] = ()
    set_at: float = field(default_factory=time.monotonic)

    def age_s(self, now: float | None = None) -> float:
        return (time.monotonic() if now is None else now) - self.set_at


def parse_intent(text: str, known_labels: Iterable[str] = ()) -> IntentProfile:
    """Map a user utterance to an intent mode + any object it names.

    Search intent is checked BEFORE navigation because "where is the door"
    names a target and should announce it when found, not merely raise
    path-hazard sensitivity.
    """
    lowered = (text or "").lower()
    words = set(re.findall(r"[a-z']+", lowered))

    targets = tuple(
        lbl for lbl in known_labels
        if lbl and lbl.lower() in lowered
    )

    has_search = any(w in lowered for w in _SEARCH_WORDS) or bool(words & {"find", "where", "locate"})
    has_nav = bool(words & _NAV_WORDS)

    if has_search and targets:
        mode = IntentMode.SEARCHING
    elif has_nav:
        mode = IntentMode.NAVIGATING
    elif has_search:
        mode = IntentMode.SEARCHING
    elif lowered.strip():
        mode = IntentMode.INSPECTING
    else:
        mode = IntentMode.UNKNOWN

    return IntentProfile(raw_text=text or "", mode=mode, target_labels=targets)


# ==========================================================================
# The blackboard itself
# ==========================================================================

class SceneState:
    """Thread-safe shared state. The detector thread writes objects; the VLM
    worker reads facts and writes intent; the orchestrator reads both.

    All public methods take the lock — callers never need to.
    """

    def __init__(
        self,
        grounding_enabled: bool = True,
        max_objects_in_prompt: int = 8,
        max_fact_age_s: float = 2.0,
        intent_navigating_boost: int = 1,
        intent_inspecting_damp: int = 1,
    ):
        self._lock = threading.RLock()
        self._objects: list[TrackedObject] = []
        self._updated_at: float = 0.0
        self._frame_shape: tuple[int, int] | None = None
        self._intent = IntentProfile()
        self._announced_targets: dict[str, float] = {}

        self.grounding_enabled = grounding_enabled
        self.max_objects_in_prompt = max_objects_in_prompt
        self.max_fact_age_s = max_fact_age_s
        self.intent_navigating_boost = intent_navigating_boost
        self.intent_inspecting_damp = intent_inspecting_damp

    # -- forward path: detector writes -------------------------------------

    def update_objects(self, objects: list[TrackedObject], frame_shape: tuple[int, int] | None = None) -> None:
        with self._lock:
            self._objects = list(objects)
            self._updated_at = time.monotonic()
            if frame_shape is not None:
                self._frame_shape = frame_shape

    def snapshot(self) -> tuple[list[TrackedObject], float]:
        """Current objects and the age in seconds of that observation."""
        with self._lock:
            age = time.monotonic() - self._updated_at if self._updated_at else float("inf")
            return list(self._objects), age

    # -- forward path: VLM reads -------------------------------------------

    def grounding_block(self, now: float | None = None) -> str:
        """The text injected into VLM prompts.

        Returns "" when grounding is disabled (the day-6 ablation) or when the
        detector's last observation is too old to be trustworthy — sending
        stale geometry as fact would trade one failure mode for another.
        """
        if not self.grounding_enabled:
            return ""
        objects, age = self.snapshot()
        if age > self.max_fact_age_s:
            return ""
        if not objects:
            return (
                "DETECTOR FACTS (ground truth from a separate object detector on this exact frame):\n"
                "- The detector found no recognisable objects in this frame.\n"
                "Note: the detector only recognises a fixed set of common object types, "
                "so an empty list does NOT prove the space is empty or safe to walk through."
            )

        ordered = sorted(objects, key=lambda o: (-int(o.severity), o.distance_m or 1e6))
        lines = [o.grounding_line() for o in ordered[: self.max_objects_in_prompt]]
        return (
            "DETECTOR FACTS (ground truth from a separate object detector on this exact frame):\n"
            + "\n".join(lines)
            + "\nThese measurements are more reliable than your visual impression of distance. "
            "If the user's statement contradicts them, correct the user."
        )

    # -- backward path: VLM writes intent ----------------------------------

    def set_intent(self, text: str) -> IntentProfile:
        known = {o.label for o in self._objects}
        # Include common COCO labels the user might name even if not currently
        # visible, so "where are my keys" still registers as a search.
        profile = parse_intent(text, known_labels=known | _COMMON_SEARCH_LABELS)
        with self._lock:
            self._intent = profile
        return profile

    def get_intent(self) -> IntentProfile:
        with self._lock:
            return self._intent

    def clear_intent(self) -> None:
        with self._lock:
            self._intent = IntentProfile()
            self._announced_targets.clear()

    # -- backward path: detector/orchestrator read salience ----------------

    def apply_salience(self, obj: TrackedObject, now: float | None = None) -> Severity:
        """Re-weight one object's severity using the current user intent.

        Rules (deliberately few and legible):
          NAVIGATING  - objects in the walking path get +1 severity: the user
                        told us they are moving, so path obstacles matter more.
          INSPECTING  - non-critical alerts get -1: the user is listening to a
                        description; chatter competes with it. CRITICAL is
                        never damped — safety is not negotiable.
          SEARCHING   - untouched here; handled by pending_target_announcement,
                        because a found object is an answer, not a hazard.
        """
        if not self.grounding_enabled:
            return obj.severity
        intent = self.get_intent()
        if intent.mode == IntentMode.NAVIGATING and obj.in_danger_zone:
            return obj.severity.bump(self.intent_navigating_boost)
        if intent.mode == IntentMode.INSPECTING and obj.severity < Severity.CRITICAL:
            return obj.severity.bump(-self.intent_inspecting_damp)
        return obj.severity

    def pending_target_announcement(
        self, cooldown_s: float = 10.0, now: float | None = None
    ) -> TrackedObject | None:
        """If the user asked to find something and it is now visible, return it
        once (per cooldown) so the proactive layer can volunteer the answer.

        This is the clearest behavioural payoff of the backward path: a purely
        geometric hazard layer would never mention a mug on a table, because a
        mug is not a hazard — but it is exactly what the user asked for.
        """
        if not self.grounding_enabled:
            return None
        now = time.monotonic() if now is None else now
        intent = self.get_intent()
        if intent.mode != IntentMode.SEARCHING or not intent.target_labels:
            return None
        objects, age = self.snapshot()
        if age > self.max_fact_age_s:
            return None
        with self._lock:
            for obj in sorted(objects, key=lambda o: o.distance_m or 1e6):
                if obj.label in intent.target_labels:
                    last = self._announced_targets.get(obj.label)
                    if last is not None and (now - last) < cooldown_s:
                        continue
                    self._announced_targets[obj.label] = now
                    return obj
        return None


# Labels a user commonly searches for that may not be on screen when they ask.
_COMMON_SEARCH_LABELS = {
    "chair", "cup", "bottle", "laptop", "backpack", "person", "door",
    "dining table", "couch", "tv", "keyboard", "mouse", "book", "cell phone",
    "remote", "handbag", "suitcase", "bench", "refrigerator",
}
