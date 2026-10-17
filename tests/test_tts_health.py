"""TTS worker failure handling.

Regression tests for a real bug: _init_engine() was called unguarded at the
top of the worker thread, so any construction failure killed the thread
outright. Every later say() queued into a queue nobody drained, and the system
went permanently mute with no error printed anywhere — the worst kind of
failure, because it is indistinguishable from "no alerts were triggered".
"""

from unittest import mock

import pipeline.tts as tts_mod
from pipeline.tts import TTSWorker


def _broken_worker() -> TTSWorker:
    with mock.patch.object(TTSWorker, "_init_engine",
                           side_effect=RuntimeError("SAPI5 unavailable")):
        return TTSWorker(rate=180).start(wait=3.0)


def test_worker_survives_engine_construction_failure():
    w = _broken_worker()
    try:
        assert w.ready.is_set(), "start() must not block forever on a broken engine"
        assert w.is_healthy is False
        assert w._thread is not None and w._thread.is_alive()
    finally:
        w.stop()


def test_failure_reason_is_reported_not_swallowed():
    w = _broken_worker()
    try:
        assert w.engine_error is not None
        assert "SAPI5 unavailable" in w.engine_error
        assert "FAILED" in w.health_line()
    finally:
        w.stop()


def test_utterances_still_drain_when_engine_is_dead(capsys):
    """A dead engine must not let the queue back up unboundedly, and the text
    must remain observable so arbitration is still demonstrable."""
    w = _broken_worker()
    try:
        w.say("Stop. person directly ahead.", priority=0, kind="alert")
        w.say("chair to your left.", priority=2, kind="alert")
        assert w.wait_until_idle(timeout=5) is True
        out = capsys.readouterr().out
        assert "[SPEAK/alert] Stop. person directly ahead." in out
    finally:
        w.stop()


def test_health_line_before_ready():
    w = TTSWorker(rate=180)          # never started
    assert "starting" in w.health_line()


def test_clear_pending_filters_by_kind():
    w = _broken_worker()
    try:
        w.stop()                      # stop draining so the queue holds items
        w.say("a", priority=5, kind="alert")
        w.say("b", priority=5, kind="reactive")
        w.say("c", priority=5, kind="alert")
        dropped = w.clear_pending(kinds=("alert",))
        assert dropped == 2
    finally:
        w.stop()
