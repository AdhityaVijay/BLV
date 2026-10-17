"""Orchestrator — the single runnable entry point. (Day 3 core deliverable.)

Ties the proactive detector layer and the reactive VLM layer together over one
camera and one audio channel, and arbitrates between them.

THREADS
-------
    capture   (in ThreadedCamera)  always holds the newest frame
    detector                       YOLO on the newest frame, writes the blackboard
    tts                            speaks from a priority queue, interruptible
    vlm worker                     one reactive query at a time, off the main loop
    main                           window + keyboard, arbitration decisions

The threading is not incidental. pyttsx3's runAndWait() blocks for the length
of the utterance; running it on the perception loop means the system is blind
for the 2-3 seconds it spends telling you about the last thing it saw. Since
the hazard it then misses is exactly the one that arrived while it was
talking, that is a safety property, not a performance nicety.

CONTROLS
--------
    SPACE   ask a question (typed in the console; enter blank to cancel)
    d       quick "what's in front of me?" without typing
    e       end the current conversation session
    q       quit

Keyboard-through-the-OpenCV-window is a developer trigger, not an accessible
one — a blind user cannot focus a window they cannot see. That is a documented
scope cut (README, Known limitations), not an oversight: speech-to-text
triggering is explicitly out of scope for this sprint.
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2

from pipeline.arbitration import Arbiter, Decision, TurnState
from pipeline.capture import ThreadedCamera
from pipeline.config_util import load_config, resolve_path
from pipeline.detector import Detector
from pipeline.hud import HudState, draw_hud
from pipeline.logging_util import EventLogger
from pipeline.prompt_library import PromptLibrary
from pipeline.scene_state import IntentMode, SceneState, Severity, TrackedObject
from pipeline.tts import SpeechGate, TTSWorker
from pipeline.vlm import VLM

# Priorities for the TTS queue (lower speaks first).
PRIORITY = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 2,
    Severity.MEDIUM: 5,
    Severity.LOW: 7,
}
REACTIVE_PRIORITY = 4     # between HIGH and MEDIUM alerts
TARGET_FOUND_PRIORITY = 3


class Orchestrator:
    def __init__(self, cfg: dict, headless: bool = False, no_grounding: bool = False,
                 hud: bool = True):
        self.cfg = cfg
        self.headless = headless
        self.show_hud = hud

        log_path = resolve_path(cfg["logging"]["session_log"])
        self.logger = EventLogger(log_path)

        g = cfg["grounding"]
        grounding_enabled = bool(g.get("enabled", True)) and not no_grounding
        self.scene = SceneState(
            grounding_enabled=grounding_enabled,
            max_objects_in_prompt=g.get("max_objects_in_prompt", 8),
            max_fact_age_s=g.get("max_fact_age_s", 2.0),
            intent_navigating_boost=g.get("intent_navigating_boost", 1),
            intent_inspecting_damp=g.get("intent_inspecting_damp", 1),
        )

        a = cfg["arbitration"]
        self.arbiter = Arbiter(
            interrupt_threshold=Severity.parse(a["interrupt_threshold"]),
            queue_threshold=Severity.parse(a["queue_threshold"]),
        )
        self.gate = SpeechGate(
            cooldown_s=a["alert_cooldown_s"],
            min_gap_s=a["min_gap_between_alerts_s"],
        )
        self.queue_max_age = a.get("queue_max_age_s", 8.0)
        # Critical hazards repeat while the danger persists, but at a pace a
        # listener can follow rather than at the detector's frame rate.
        self.critical_cooldown_s = a.get("critical_cooldown_s", 2.5)

        self.library = PromptLibrary()
        self.hazard_templates = self.library.by_category("hazard_alert").templates

        self.detector = Detector(cfg, logger=self.logger)
        self.vlm = VLM(cfg, scene_state=self.scene, logger=self.logger, library=self.library)
        self.tts = TTSWorker(
            rate=cfg["tts"]["rate"],
            volume=cfg["tts"]["volume"],
            on_state_change=self._on_speech_state,
        )
        self.camera = ThreadedCamera(
            device_index=cfg["camera"]["device_index"],
            width=cfg["camera"]["frame_width"],
            height=cfg["camera"]["frame_height"],
            fps=cfg["camera"]["fps"],
        )

        self._state = TurnState.IDLE
        self._state_lock = threading.RLock()
        self._deferred: list[tuple[float, TrackedObject, Severity]] = []
        self._stop = threading.Event()
        self._vlm_busy = threading.Event()
        self._grounding_enabled = grounding_enabled

        # Live display state. Written by the detector/VLM threads, read by the
        # main loop for the HUD - all small immutable values, so a plain
        # assignment is atomic enough and no extra lock is warranted.
        self._last_alert_text = ""
        self._last_alert_severity: Severity | None = None
        self._speaking_alert_severity: Severity | None = None
        self._last_decision_by_key: dict[str, str] = {}
        self._prompting = threading.Event()
        self._last_answer = ""
        self._last_answer_ms: float | None = None
        self._detect_ms = 0.0
        self._fps = 0.0
        self._detect_errors = 0

    # -- state -------------------------------------------------------------

    def _set_state(self, state: TurnState) -> None:
        with self._state_lock:
            self._state = state

    def _get_state(self) -> TurnState:
        with self._state_lock:
            return self._state

    def _on_speech_state(self, kind: str, speaking: bool) -> None:
        """TTS worker callback — keeps the arbiter's view of the audio channel
        accurate without the main loop having to poll it."""
        if speaking:
            self._set_state(
                TurnState.REACTIVE_SPEAKING if kind == "reactive" else TurnState.ALERT_SPEAKING
            )
        else:
            if kind == "alert":
                self._speaking_alert_severity = None
            # Only drop to IDLE if no reactive query is still pending.
            if kind == "reactive" or not self._vlm_busy.is_set():
                self._set_state(TurnState.IDLE)
                self._flush_deferred()
            else:
                self._set_state(TurnState.REACTIVE_PENDING)

    # -- proactive path ----------------------------------------------------

    def _phrase_for(self, obj: TrackedObject, severity: Severity) -> str:
        template = self.hazard_templates.get(severity.name.lower(), "{label} {position_phrase}.")
        return template.format(
            label=obj.label,
            position_phrase=obj.spoken_position(),
            distance_phrase=obj.spoken_distance(),
        )

    def _handle_hazard(self, obj: TrackedObject) -> None:
        # BACKWARD PATH of the blackboard: intent re-weights severity before
        # the arbiter ever sees it.
        effective = self.scene.apply_salience(obj)
        state = self._get_state()
        is_critical = effective >= Severity.CRITICAL
        gate_ok = self.gate.allows(
            obj.key,
            bypass_gap=is_critical,
            cooldown_override=self.critical_cooldown_s if is_critical else None,
        )
        result = self.arbiter.decide(
            effective, state, gate_ok,
            speaking_severity=self._speaking_alert_severity,
        )

        # Log the decision ONCE, and only when it changes for this object.
        # The detector runs at ~10 Hz, so logging every tick buried 7 real
        # alerts under 400 lines of identical suppressions and made the
        # arbitration record unreadable as evidence.
        prev = self._last_decision_by_key.get(obj.key)
        if prev != result.decision.value:
            self._last_decision_by_key[obj.key] = result.decision.value
            self.logger.log(
                "arbitration",
                decision=result.decision.value,
                reason=result.reason,
                object_class=obj.label,
                position=obj.position,
                distance_m=round(obj.distance_m, 2) if obj.distance_m is not None else None,
                base_severity=obj.severity.name,
                effective_severity=effective.name,
                intent_mode=self.scene.get_intent().mode,
                state=state.value,
            )

        if result.decision == Decision.SUPPRESSED:
            return
        if result.decision == Decision.QUEUED:
            self._deferred.append((time.monotonic(), obj, effective))
            return

        phrase = self._phrase_for(obj, effective)
        if result.decision == Decision.INTERRUPTED:
            # Cut off whatever is speaking, and drop queued non-critical items
            # so the user isn't buried in backlog after a safety interrupt.
            # The "don't preempt an equally urgent alert" rule now lives in
            # the arbiter, so anything reaching here genuinely should cut in.
            self.tts.interrupt()
            self.tts.clear_pending(kinds=("alert",))

        self._speaking_alert_severity = effective
        self.tts.say(phrase, priority=PRIORITY[effective], kind="alert")
        self.gate.mark(obj.key)
        self._last_alert_text = phrase
        self._last_alert_severity = effective
        self.logger.log(
            "alert",
            object_class=obj.label,
            position=obj.position,
            distance_m=round(obj.distance_m, 2) if obj.distance_m is not None else None,
            severity=effective.name,
            text=phrase,
            decision=result.decision.value,
        )

    def _flush_deferred(self) -> None:
        """Speak alerts that were queued during a reactive turn — but drop any
        that have gone stale. A hazard observed eight seconds ago is no longer
        a description of the world the user is standing in."""
        if not self._deferred:
            return
        now = time.monotonic()
        pending, self._deferred = self._deferred, []
        for queued_at, obj, severity in pending:
            age = now - queued_at
            if age > self.queue_max_age:
                self.logger.log(
                    "arbitration", decision="suppressed",
                    reason=f"queued alert went stale ({age:.1f}s > {self.queue_max_age}s)",
                    object_class=obj.label, effective_severity=severity.name,
                    state=self._get_state().value,
                )
                continue
            if not self.gate.allows(obj.key):
                continue
            phrase = self._phrase_for(obj, severity)
            self.tts.say(phrase, priority=PRIORITY[severity], kind="alert")
            self.gate.mark(obj.key)
            self.logger.log("alert", object_class=obj.label, position=obj.position,
                            severity=severity.name, text=phrase, decision="flushed_from_queue")

    def _check_target_announcement(self) -> None:
        """Backward path payoff: the user asked to find something, and it is
        now in view. A purely geometric hazard layer would never mention a mug;
        this does, because the conversation told it to care."""
        cooldown = self.cfg["grounding"].get("search_announce_cooldown_s", 10.0)
        obj = self.scene.pending_target_announcement(cooldown_s=cooldown)
        if obj is None:
            return
        if self._get_state().reactive_in_flight:
            return
        template = self.hazard_templates.get("target_found", "I can see the {label} now.")
        phrase = template.format(
            label=obj.label,
            position_phrase=obj.spoken_position(),
            distance_phrase=obj.spoken_distance(),
        )
        self.tts.say(phrase, priority=TARGET_FOUND_PRIORITY, kind="alert")
        self.logger.log("alert", object_class=obj.label, position=obj.position,
                        severity="TARGET_FOUND", text=phrase, decision="intent_driven",
                        intent_mode=IntentMode.SEARCHING)

    # -- detector thread ---------------------------------------------------

    def _detector_loop(self) -> None:
        every_n = self.cfg["camera"].get("throttle_every_n_frames", 3)
        interval = every_n / float(self.cfg["camera"].get("fps", 30) or 30)
        while not self._stop.is_set():
            frame, age = self.camera.read_with_age()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                t0 = time.perf_counter()
                objects = self.detector.detect(frame)
                self._detect_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                # Print, don't only log. A detector throwing on every frame
                # used to be invisible on screen: no boxes, no alerts, no
                # error - identical to "nothing is in view".
                self._detect_errors += 1
                if self._detect_errors <= 3:
                    print(f"[detector] inference failed ({type(e).__name__}: {e})")
                    if self._detect_errors == 3:
                        print("[detector] further inference errors will be logged only")
                self.logger.log("system", component="detector", error=str(e))
                time.sleep(0.2)
                continue
            self.scene.update_objects(objects, frame_shape=frame.shape[:2])
            self._check_target_announcement()
            for obj in objects:
                if obj.severity >= Severity.MEDIUM or self.scene.apply_salience(obj) >= Severity.MEDIUM:
                    self._handle_hazard(obj)
            time.sleep(interval)

    # -- reactive path -----------------------------------------------------

    def _ask_async(self, question: str, category: str = "scene_description") -> None:
        if self._vlm_busy.is_set():
            print("[orchestrator] a question is already in flight; ignoring.")
            return

        def worker() -> None:
            self._vlm_busy.set()
            self._set_state(TurnState.REACTIVE_PENDING)
            frame = self.camera.read()
            try:
                resp = self.vlm.ask(frame, question, category=category)
                if resp.error:
                    print(f"[vlm] error: {resp.error}")
                    self.tts.say("Sorry, I couldn't reach the vision model.",
                                 priority=REACTIVE_PRIORITY, kind="reactive")
                else:
                    print(f"[vlm] ({resp.latency_ms:.0f} ms, grounded={resp.grounded}) {resp.text}")
                    self._last_answer = resp.text
                    self._last_answer_ms = resp.latency_ms
                    self.tts.say(resp.text, priority=REACTIVE_PRIORITY, kind="reactive")
            finally:
                self._vlm_busy.clear()
                if not self.tts.is_speaking:
                    self._set_state(TurnState.IDLE)
                    self._flush_deferred()

        threading.Thread(target=worker, name="vlm-query", daemon=True).start()

    def _prompt_async(self) -> None:
        """Read a typed question WITHOUT blocking the main loop.

        input() used to be called inline, which froze the video window, the
        keyboard handler and the whole display until a line was typed in the
        terminal. Since the keys are pressed in the video window and the text
        is typed in the terminal, the obvious thing to do - press space, look
        at the window - left the app looking completely hung.
        """
        if self._prompting.is_set():
            return
        self._prompting.set()

        def reader() -> None:
            try:
                print("\n>>> Type your question in THIS terminal, then Enter "
                      "(blank cancels):")
                question = input("    question> ").strip()
                if not question:
                    print("    cancelled.")
                    return
                lowered = question.lower()
                category = ("object_localization"
                            if any(w in lowered for w in ("find", "where", "locate"))
                            else "scene_description")
                self._ask_async(question, category=category)
            except (EOFError, KeyboardInterrupt):
                pass
            finally:
                self._prompting.clear()

        threading.Thread(target=reader, name="prompt", daemon=True).start()

    # -- display -----------------------------------------------------------

    def _render_hud(self, frame):
        """Assemble the live display from current blackboard state."""
        objects, age = self.scene.snapshot()
        intent = self.scene.get_intent()
        # Only draw boxes from a fresh observation; stale boxes drifting over a
        # moving scene would misrepresent what the system currently believes.
        fresh = objects if age <= self.scene.max_fact_age_s else []
        state = HudState(
            grounding_enabled=self._grounding_enabled,
            turn_state=self._get_state().value,
            intent_mode=intent.mode,
            intent_targets=intent.target_labels,
            grounding_block=self.scene.grounding_block(),
            last_alert=self._last_alert_text,
            last_alert_severity=self._last_alert_severity,
            last_answer=self._last_answer,
            last_answer_ms=self._last_answer_ms,
            fps=self._fps,
            detect_ms=self._detect_ms,
            queued_alerts=len(self._deferred),
            objects=fresh,
        )
        return draw_hud(frame, state, danger_zone=self.cfg["detector"]["danger_zone"])

    # -- run ---------------------------------------------------------------

    def run(self) -> None:
        print("=" * 68)
        print("BLV assistant — orchestrator")
        print(f"  grounding (detector->VLM facts, VLM->salience): "
              f"{'ENABLED' if self._grounding_enabled else 'DISABLED (ablation)'}")
        print(f"  interrupt threshold: {self.arbiter.interrupt_threshold.name}  "
              f"queue threshold: {self.arbiter.queue_threshold.name}")
        avail = self.vlm.check_available()
        if avail.get("ok"):
            print(f"  ollama: {avail['active_model']} @ {avail['host']}")
        else:
            print(f"  ollama: UNAVAILABLE — {avail.get('note') or avail.get('error')}")
            print("         proactive hazard alerts will still work.")
        print("  keys: SPACE=ask   d=describe   e=end conversation   q=quit")
        print("  >>> CLICK THE VIDEO WINDOW FIRST - keys are read there, not here.")
        print("      (SPACE then switches you back here to type the question.)")
        print("=" * 68)

        self.logger.log("system", component="orchestrator", event="start",
                        grounding_enabled=self._grounding_enabled)

        self.tts.start()
        # Surface speech health at startup. A mute session should announce
        # itself in the first second, not be discovered twenty minutes in.
        print(f"  {self.tts.health_line()}")
        if self.tts.is_healthy:
            self.tts.say("B L V assistant ready.", priority=9, kind="alert")
        else:
            print("  run 'python -m pipeline.check_tts' to diagnose; "
                  "alerts will print as [SPEAK] lines meanwhile")
        self.logger.log("system", component="tts",
                        healthy=self.tts.is_healthy, error=self.tts.engine_error)

        self.camera.open()
        det_thread = threading.Thread(target=self._detector_loop, name="detector", daemon=True)
        det_thread.start()

        last_frame_t = time.perf_counter()
        try:
            while not self._stop.is_set():
                frame = self.camera.read()
                if frame is None:
                    time.sleep(0.05)
                    continue

                if not self.headless:
                    if self.show_hud:
                        cv2.imshow("BLV assistant", self._render_hud(frame))
                    else:
                        cv2.imshow("BLV assistant", frame)
                    key = cv2.waitKey(30) & 0xFF
                    # Exponentially smoothed FPS - an instantaneous reading
                    # jitters too much to read off a live display.
                    now = time.perf_counter()
                    dt = now - last_frame_t
                    last_frame_t = now
                    if dt > 0:
                        self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
                else:
                    key = 0xFF
                    time.sleep(0.05)

                if key == ord("q"):
                    break
                elif key == ord("d"):
                    self._ask_async("What's in front of me?", category="scene_description")
                elif key == ord("e"):
                    self.vlm.end_session()
                    print("[orchestrator] conversation session ended.")
                elif key == ord(" "):
                    self._prompt_async()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        self.logger.log("system", component="orchestrator", event="stop")
        try:
            self.tts.stop()
        except Exception:
            pass
        self.camera.release()
        if not self.headless:
            cv2.destroyAllWindows()
        self.logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BLV assistant orchestrator")
    parser.add_argument("--headless", action="store_true",
                        help="run without the preview window (no keyboard triggers)")
    parser.add_argument("--no-grounding", action="store_true",
                        help="disable the scene-state grounding path (day-6 ablation)")
    parser.add_argument("--no-hud", action="store_true",
                        help="plain camera preview instead of the live state HUD")
    args = parser.parse_args()

    cfg = load_config()
    Orchestrator(
        cfg,
        headless=args.headless,
        no_grounding=args.no_grounding,
        hud=not args.no_hud,
    ).run()


if __name__ == "__main__":
    main()
