"""Text-to-speech wrapper.

Three pieces:

  speak(text)  - module-level convenience for the simple day-1/day-2 scripts.

  SpeechGate   - cooldown / de-duplication. Stops the same alert repeating on
                 every frame, and enforces a floor on the gap between any two
                 utterances. Pure logic, unit tested.

  TTSWorker    - a speech thread with a priority queue and interruption
                 support. The orchestrator needs this: pyttsx3's runAndWait()
                 is synchronous, so calling it on the main loop freezes capture
                 and detection for the whole utterance. Speech runs on its own
                 thread, and a CRITICAL hazard can call interrupt() to cut off
                 an in-progress sentence.

Windows note: SAPI5 is COM-based and thread-affine. The pyttsx3 engine is
therefore created INSIDE the worker thread, with CoInitialize() called first.
Creating it on the main thread and using it from the worker silently fails or
deadlocks on Windows.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

# NOTE: pyttsx3 is imported lazily inside the functions that actually speak.
# SpeechGate is pure logic, and importing it must not require an audio stack —
# otherwise the cooldown rules can't be unit tested on a CI runner or any
# machine without a speech engine installed.


# --------------------------------------------------------------------------
# Cooldown / de-duplication
# --------------------------------------------------------------------------

@dataclass
class SpeechGate:
    """Decides whether an utterance is allowed to be spoken right now."""

    cooldown_s: float = 6.0        # per-key: same phrase won't repeat within this
    min_gap_s: float = 1.5         # global: floor between any two utterances
    _last_by_key: dict[str, float] = field(default_factory=dict)
    _last_any: float | None = None

    def allows(
        self,
        key: str,
        now: float | None = None,
        bypass_gap: bool = False,
        cooldown_override: float | None = None,
    ) -> bool:
        """Two independent checks, and the distinction matters.

        bypass_gap=True skips only the GLOBAL spacing rule, so a critical
        hazard is never withheld merely because something else spoke a moment
        ago. It does NOT skip the per-key cooldown.

        That separation is a bug fix, not a nicety. Critical alerts previously
        bypassed both, so a persistent hazard - a person simply standing in
        frame - re-fired at the detector's rate of roughly ten times a second.
        Each repeat preempted the one before it, so no utterance ever reached
        the end of its sentence and the system was silent while looking, in
        the logs, like it was alerting constantly.

        cooldown_override lets critical hazards use a SHORTER per-key cooldown
        than routine ones: repeating "stop, person ahead" every couple of
        seconds while you remain in danger is correct; repeating it ten times
        a second is not speech.
        """
        now = time.monotonic() if now is None else now
        cooldown = self.cooldown_s if cooldown_override is None else cooldown_override

        if not bypass_gap and self._last_any is not None:
            if (now - self._last_any) < self.min_gap_s:
                return False

        last = self._last_by_key.get(key)
        if last is not None and (now - last) < cooldown:
            return False
        return True

    def mark(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._last_by_key[key] = now
        self._last_any = now

    def reset(self) -> None:
        self._last_by_key.clear()
        self._last_any = None


# --------------------------------------------------------------------------
# Simple synchronous speak (day 1 / day 2 scripts)
# --------------------------------------------------------------------------

_engine = None
_engine_lock = threading.Lock()


def speak(text: str, rate: int = 185, volume: float = 1.0) -> None:
    """Blocking speak. Fine for the standalone scripts; the orchestrator uses
    TTSWorker instead so speech never stalls the perception loop."""
    import pyttsx3

    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = pyttsx3.init()
        _engine.setProperty("rate", rate)
        _engine.setProperty("volume", volume)
        _engine.say(text)
        _engine.runAndWait()


# --------------------------------------------------------------------------
# Threaded worker with interruption (orchestrator)
# --------------------------------------------------------------------------

@dataclass(order=True)
class _Utterance:
    # Lower priority value = spoken sooner. Ties break by sequence number so
    # equal-priority utterances stay FIFO.
    priority: int
    seq: int
    text: str = field(compare=False)
    kind: str = field(compare=False, default="alert")


class TTSWorker:
    def __init__(self, rate: int = 185, volume: float = 1.0, on_state_change=None):
        self.rate = rate
        self.volume = volume
        self._q: queue.PriorityQueue[_Utterance] = queue.PriorityQueue()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stop = threading.Event()
        self._interrupt = threading.Event()
        self._speaking = threading.Event()
        self._engine = None
        self._thread: threading.Thread | None = None
        self._current_kind: str | None = None
        self.on_state_change = on_state_change  # callback(kind, is_speaking)

        # Health signals. Without these a broken engine is indistinguishable
        # from silence, which is the exact failure this class had.
        self.ready = threading.Event()
        self._engine_error: str | None = None
        self._speak_errors = 0
        self._spoken_count = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self, wait: float = 5.0) -> "TTSWorker":
        """Start the worker and wait for the engine to come up (or fail).

        Waiting matters: callers need to know whether audio works BEFORE the
        session starts, not after wondering for ten minutes why it is quiet.
        """
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()
        self.ready.wait(timeout=wait)
        return self

    @property
    def engine_error(self) -> str | None:
        """None when speech is working, else why it isn't."""
        return self._engine_error

    @property
    def is_healthy(self) -> bool:
        return self._engine is not None and self._engine_error is None

    def health_line(self) -> str:
        if self.is_healthy:
            return "speech: OK"
        if not self.ready.is_set():
            return "speech: still starting"
        return f"speech: FAILED ({self._engine_error}) — text will print instead"

    def stop(self) -> None:
        self._stop.set()
        self._interrupt.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # -- queue API ---------------------------------------------------------

    def say(self, text: str, priority: int = 5, kind: str = "alert") -> None:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        self._q.put(_Utterance(priority=priority, seq=seq, text=text, kind=kind))

    def interrupt(self) -> None:
        """Cut off the current utterance (used by CRITICAL hazards)."""
        self._interrupt.set()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def clear_pending(self, kinds: tuple[str, ...] | None = None) -> int:
        """Drop queued utterances, optionally only those of given kinds.
        Returns how many were dropped."""
        kept: list[_Utterance] = []
        dropped = 0
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if kinds is None or item.kind in kinds:
                dropped += 1
            else:
                kept.append(item)
        for item in kept:
            self._q.put(item)
        return dropped

    # -- state -------------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def current_kind(self) -> str | None:
        return self._current_kind

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty() and not self.is_speaking:
                return True
            time.sleep(0.05)
        return False

    # -- worker ------------------------------------------------------------

    def _init_engine(self) -> None:
        import pyttsx3

        # Windows/SAPI5: COM must be initialised on this thread before pyttsx3.
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
        except Exception:
            pass
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", self.rate)
        self._engine.setProperty("volume", self.volume)

    def _run(self) -> None:
        # Engine construction MUST be guarded. It previously was not, and an
        # exception here killed the worker thread on the spot: every later
        # say() then queued into a queue nobody was draining, so the system
        # went permanently, silently mute with no error anywhere. On Windows
        # this is a live risk, because SAPI5 is COM-based and pyttsx3.init()
        # fails on a non-main thread if CoInitialize() did not run.
        try:
            self._init_engine()
            self.ready.set()
        except Exception as e:
            self._engine_error = f"{type(e).__name__}: {e}"
            self._engine = None
            self.ready.set()   # unblock waiters; they check engine_error
            print("\n" + "!" * 68)
            print(f"[tts] SPEECH ENGINE FAILED TO START: {self._engine_error}")
            print("[tts] Falling back to printing what would have been spoken.")
            print("[tts] Diagnose with:  python -m pipeline.check_tts")
            print("!" * 68 + "\n")

        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # No engine: keep the pipeline observable rather than dropping
            # utterances on the floor. Arbitration, logging and the HUD all
            # still work; only the audio is missing.
            if self._engine is None:
                print(f"[SPEAK/{item.kind}] {item.text}")
                self._spoken_count += 1
                continue

            self._interrupt.clear()
            self._speaking.set()
            self._current_kind = item.kind
            if self.on_state_change:
                try:
                    self.on_state_change(item.kind, True)
                except Exception:
                    pass
            try:
                self._engine.say(item.text)
                self._engine.runAndWait()
                self._spoken_count += 1
            except RuntimeError:
                # pyttsx3 raises if the loop is already running after a stop();
                # re-init the engine and drop this utterance rather than die.
                try:
                    self._init_engine()
                except Exception as e:
                    self._engine_error = f"re-init failed: {type(e).__name__}: {e}"
                    self._engine = None
                    print(f"[tts] {self._engine_error} — falling back to console output")
            except Exception as e:
                # Previously a bare `pass`, which hid every speech failure.
                # Report the first few, then stop repeating ourselves.
                self._speak_errors += 1
                if self._speak_errors <= 3:
                    print(f"[tts] speak failed ({type(e).__name__}: {e})")
                    print(f"[SPEAK/{item.kind}] {item.text}")
            finally:
                self._speaking.clear()
                kind, self._current_kind = self._current_kind, None
                if self.on_state_change:
                    try:
                        self.on_state_change(kind or item.kind, False)
                    except Exception:
                        pass
