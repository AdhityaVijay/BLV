"""Decides WHETHER and WHEN to speak a detection, so the user isn't
bombarded with "person ahead" every single frame. Pure logic, no I/O —
easy to unit test (see tests/test_alert_manager.py).

Two cooldowns:
  - per-key cooldown: the same (label, position) phrase won't repeat
    within `alert_cooldown_seconds`.
  - global gap: any two spoken alerts are at least
    `min_priority_gap_seconds` apart, so overlapping detections queue
    up sensibly instead of talking over each other.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AlertManager:
    alert_cooldown_seconds: float = 4.0
    min_priority_gap_seconds: float = 0.8
    _last_spoken_at: dict[str, float] = field(default_factory=dict)
    _last_any_alert_at: float | None = field(default=None)

    def should_speak(self, key: str, now: float | None = None) -> bool:
        """key is typically f"{label}:{position}", e.g. 'person:left'."""
        now = now if now is not None else time.monotonic()

        if self._last_any_alert_at is not None and (now - self._last_any_alert_at) < self.min_priority_gap_seconds:
            return False

        last = self._last_spoken_at.get(key)
        if last is not None and (now - last) < self.alert_cooldown_seconds:
            return False

        return True

    def mark_spoken(self, key: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._last_spoken_at[key] = now
        self._last_any_alert_at = now
