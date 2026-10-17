"""Live heads-up display: draws what the system is thinking, next to what it
is seeing.

Why this exists beyond looking nice: the contribution of this project is a
shared blackboard between the detector and the language model, and that is
invisible in a plain camera preview. Printing it to a console scrolls past too
fast to follow. Rendering the DETECTOR FACTS block live, beside the boxes those
facts came from, next to the intent the conversation wrote back, makes the
coupling something you can point at — in debugging and in a demo.

Drawing is kept out of orchestrator.py so the control flow there stays about
arbitration, and so the text-layout helpers can be unit tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from pipeline.scene_state import Severity, TrackedObject

PANEL_WIDTH = 430
FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR. Deliberately ordered green -> red so severity reads at a glance without
# needing the label.
SEVERITY_COLORS: dict[Severity, tuple[int, int, int]] = {
    Severity.LOW: (110, 200, 110),
    Severity.MEDIUM: (80, 220, 240),
    Severity.HIGH: (60, 150, 255),
    Severity.CRITICAL: (70, 70, 255),
}

BG = (26, 22, 20)
FG = (235, 235, 235)
DIM = (150, 150, 150)
ACCENT = (240, 190, 90)
GOOD = (120, 210, 120)
WARN = (70, 70, 255)


def severity_color(severity: Severity) -> tuple[int, int, int]:
    return SEVERITY_COLORS.get(severity, DIM)


def wrap_text(text: str, width: int) -> list[str]:
    """Greedy word wrap to `width` characters.

    Words longer than the width are hard-split rather than allowed to overflow
    the panel, because a single long token (a URL, a run-on model output)
    would otherwise push text off the edge of the window silently.
    """
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for word in str(text).split():
        while len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:width])
            word = word[width:]
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


@dataclass
class HudState:
    """Everything the panel renders. The orchestrator fills this each frame."""

    grounding_enabled: bool = True
    turn_state: str = "idle"
    intent_mode: str = "unknown"
    intent_targets: tuple[str, ...] = ()
    grounding_block: str = ""
    last_alert: str = ""
    last_alert_severity: Severity | None = None
    last_answer: str = ""
    last_answer_ms: float | None = None
    fps: float = 0.0
    detect_ms: float = 0.0
    queued_alerts: int = 0
    objects: list[TrackedObject] = field(default_factory=list)


def _draw_danger_zone(frame: np.ndarray, zone: dict) -> None:
    """Outline the walking corridor so it's obvious why an object in the
    middle of the frame is treated differently from one at the edge."""
    h, w = frame.shape[:2]
    x1, x2 = int(zone["x_min"] * w), int(zone["x_max"] * w)
    y1 = int(zone["y_min"] * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, h - 1), (90, 90, 90), -1)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, h - 1), (120, 120, 120), 1)
    cv2.putText(frame, "walking path", (x1 + 4, y1 + 14), FONT, 0.36, (170, 170, 170), 1, cv2.LINE_AA)


def _draw_boxes(frame: np.ndarray, objects: list[TrackedObject]) -> None:
    for obj in objects:
        x1, y1, x2, y2 = (int(v) for v in obj.bbox)
        color = severity_color(obj.severity)
        thickness = 3 if obj.severity >= Severity.HIGH else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        dist = f"{obj.distance_m:.1f}m" if obj.distance_m is not None else "?m"
        label = f"{obj.label} {dist} {obj.severity.name}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.42, 1)
        ly = max(th + 4, y1)
        cv2.rectangle(frame, (x1, ly - th - 5), (x1 + tw + 6, ly + 2), color, -1)
        cv2.putText(frame, label, (x1 + 3, ly - 3), FONT, 0.42, (20, 20, 20), 1, cv2.LINE_AA)


def _panel(state: HudState, height: int) -> np.ndarray:
    panel = np.full((height, PANEL_WIDTH, 3), BG, dtype=np.uint8)
    y = 24
    chars = 52

    def line(text: str, color=FG, scale=0.42, gap=16) -> None:
        nonlocal y
        if y < height - 4:
            cv2.putText(panel, text, (12, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += gap

    def rule() -> None:
        nonlocal y
        if y < height - 4:
            cv2.line(panel, (12, y - 6), (PANEL_WIDTH - 12, y - 6), (60, 56, 54), 1)
        y += 8

    line("BLV ASSISTANT", ACCENT, 0.5, 20)
    g_on = state.grounding_enabled
    line(f"grounding: {'ON' if g_on else 'OFF (ablation)'}", GOOD if g_on else WARN, 0.4)
    rule()

    # -- live status -------------------------------------------------------
    state_color = WARN if "speaking" in state.turn_state else DIM
    line(f"state:  {state.turn_state}", state_color)
    targets = f"  targets: {', '.join(state.intent_targets)}" if state.intent_targets else ""
    line(f"intent: {state.intent_mode}{targets}", ACCENT if state.intent_mode != "unknown" else DIM)
    q = f"   queued: {state.queued_alerts}" if state.queued_alerts else ""
    line(f"{state.fps:.0f} fps   detect {state.detect_ms:.0f} ms{q}", DIM, 0.38)
    rule()

    # -- the novel bit, rendered live --------------------------------------
    line("DETECTOR FACTS -> VLM", ACCENT, 0.4)
    if not g_on:
        line("(grounding disabled)", DIM, 0.38)
    elif not state.grounding_block:
        line("(no fresh facts)", DIM, 0.38)
    else:
        for raw in state.grounding_block.splitlines():
            raw = raw.strip()
            # Skip the prose header/footer; the fact lines are the point here.
            if not raw.startswith("-"):
                continue
            for i, wrapped in enumerate(wrap_text(raw, chars)):
                line(("  " + wrapped) if i else wrapped, FG, 0.38, 14)
    rule()

    # -- last spoken alert -------------------------------------------------
    line("LAST ALERT", ACCENT, 0.4)
    if state.last_alert:
        col = severity_color(state.last_alert_severity) if state.last_alert_severity else FG
        for wrapped in wrap_text(state.last_alert, chars):
            line(wrapped, col, 0.38, 14)
    else:
        line("(none yet)", DIM, 0.38)
    rule()

    # -- last conversational answer ----------------------------------------
    ms = f"  ({state.last_answer_ms:.0f} ms)" if state.last_answer_ms else ""
    line(f"LAST ANSWER{ms}", ACCENT, 0.4)
    if state.last_answer:
        for wrapped in wrap_text(state.last_answer, chars)[:6]:
            line(wrapped, FG, 0.38, 14)
    else:
        line("(press d or SPACE)", DIM, 0.38)

    # -- keys, pinned to the bottom ----------------------------------------
    cv2.line(panel, (12, height - 30), (PANEL_WIDTH - 12, height - 30), (60, 56, 54), 1)
    cv2.putText(panel, "SPACE ask   d describe   e end   q quit",
                (12, height - 12), FONT, 0.38, DIM, 1, cv2.LINE_AA)
    return panel


def draw_hud(frame: np.ndarray, state: HudState, danger_zone: dict | None = None) -> np.ndarray:
    """Return a new canvas: annotated video on the left, state panel on right."""
    view = frame.copy()
    if danger_zone:
        _draw_danger_zone(view, danger_zone)
    _draw_boxes(view, state.objects)

    h = view.shape[0]
    canvas = np.hstack([view, _panel(state, h)])
    return canvas
