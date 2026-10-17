"""Structured JSONL event logging — one line per pipeline event.

Requirement carried from day 1: EVERY pipeline event is logged with a
timestamp, its latency where meaningful, and its content. This file is the
evidence trail that day 7's analyze_logs.py reads, so the schema matters:

    {"ts": ISO8601, "t_mono": float, "type": <event type>, ...fields}

Event types in use:
    detection    - one detected object on one frame
    alert        - a hazard alert actually spoken
    query        - a reactive VLM round trip (has session_id, latency_ms)
    arbitration  - a turn-taking decision (queued|interrupted|allowed|suppressed)
    intent       - parsed user intent written back to the blackboard
    task         - dogfooding task start/end markers (day 7)
    redteam      - adversarial probe result (day 6)
    system       - startup/shutdown/device info
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """Thread-safe append-only JSONL writer.

    Several threads (detector, TTS worker, VLM worker) log concurrently, so
    writes are mutex-guarded and flushed per line — a crash mid-session must
    not lose the events leading up to it.
    """

    def __init__(self, path: str | Path, echo: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")
        self.echo = echo

    def log(self, event_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "t_mono": round(time.monotonic(), 4),
            "type": event_type,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
        if self.echo:
            print(f"[log] {line}")
        return record

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
