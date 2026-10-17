"""Ollama vision-LLM wrapper — reactive conversational layer.

Days 2 and 4 combined:
  * single image + prompt -> text response, with latency measured (day 2)
  * multi-turn conversation with session scoping and a history cap (day 4)

Plus the forward half of the novel grounding path: detector measurements from
the scene-state blackboard are appended to each user turn as a DETECTOR FACTS
block, and the system prompt (see prompts/system_prompts.yaml
"grounded_contract") instructs the model to trust them over the user's
assertions. Setting grounding.enabled=false in config.yaml disables this,
which is what the day-6 ablation compares against.

Image handling: minicpm-v, like the other Ollama vision models, is stateless
with respect to images — it does not retain an image across turns. The current
frame is therefore re-sent with every turn, attached to the latest user
message only. Attaching images to historical turns as well would multiply
prompt-processing time for no benefit.
"""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pipeline.prompt_library import PromptLibrary
from pipeline.scene_state import Severity, SceneState

# cv2/numpy are imported lazily: ConversationSession is pure logic and must be
# unit testable without an OpenCV install.
if TYPE_CHECKING:
    import numpy as np


def encode_frame_jpeg(frame: "np.ndarray", quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode frame for the VLM.")
    return buf.tobytes()


@dataclass
class VLMResponse:
    text: str
    latency_ms: float
    model: str
    session_id: str
    turn_index: int
    grounded: bool
    prompt_id: str
    override_applied: bool = False
    error: str | None = None


@dataclass
class ConversationSession:
    """One conversation. Starts on the first query, ends after
    session_idle_timeout_s of inactivity or an explicit end.

    History cap: only the last `max_turns` turns (a turn = one user message
    plus one assistant reply) are kept. Unbounded history would grow prompt
    processing time linearly, and on an 8 GB card that shows up as latency
    creep within a handful of exchanges.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    max_turns: int = 6
    idle_timeout_s: float = 45.0
    messages: list[dict] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)
    turn_count: int = 0

    def is_expired(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return (now - self.last_activity) > self.idle_timeout_s

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.last_activity = time.monotonic()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.turn_count += 1
        self.last_activity = time.monotonic()
        self._trim()

    def _trim(self) -> None:
        """Keep the most recent max_turns*2 messages (user+assistant pairs)."""
        limit = self.max_turns * 2
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]

    def history_for_request(self) -> list[dict]:
        return [dict(m) for m in self.messages]


class VLM:
    def __init__(self, cfg: dict, scene_state: SceneState | None = None, logger=None,
                 library: PromptLibrary | None = None):
        v = cfg["vlm"]
        self.cfg = cfg
        self.model = v["model"]
        self.fallback_model = v["fallback_model"]
        self.host = v.get("host") or "http://localhost:11434"
        self.timeout = v.get("request_timeout_s", 120)
        self.jpeg_quality = v.get("jpeg_quality", 85)
        self.max_history_turns = v.get("max_history_turns", 6)
        self.idle_timeout = v.get("session_idle_timeout_s", 45.0)

        self.scene_state = scene_state
        self.logger = logger
        self.library = library or PromptLibrary()
        self.session: ConversationSession | None = None
        self._active_model = self.model
        self._client = None

    # -- client ------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    def check_available(self) -> dict:
        """Verify Ollama is up and report which of the configured models are
        pulled. Called at startup so a missing model fails loudly and early
        rather than on the user's first question."""
        info: dict = {"host": self.host, "ok": False, "models": [], "active_model": None}
        try:
            client = self._get_client()
            listing = client.list()
            names = []
            for m in listing.get("models", []):
                name = m.get("model") or m.get("name") or ""
                if name:
                    names.append(name)
            info["models"] = names
            base = {n.split(":")[0] for n in names}
            if self.model.split(":")[0] in base:
                self._active_model = self.model
            elif self.fallback_model.split(":")[0] in base:
                self._active_model = self.fallback_model
                info["note"] = f"{self.model} not found; using fallback {self.fallback_model}"
            else:
                info["note"] = (
                    f"Neither {self.model} nor {self.fallback_model} is pulled. "
                    f"Run: ollama pull {self.model}"
                )
                info["active_model"] = None
                if self.logger:
                    self.logger.log("system", component="vlm", **info)
                return info
            info["ok"] = True
            info["active_model"] = self._active_model
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        if self.logger:
            self.logger.log("system", component="vlm", **info)
        return info

    # -- sessions ----------------------------------------------------------

    def get_or_start_session(self, now: float | None = None) -> ConversationSession:
        """Consecutive queries inside the idle window continue one conversation;
        a query after the window starts a fresh one."""
        if self.session is None or self.session.is_expired(now):
            self.session = ConversationSession(
                max_turns=self.max_history_turns,
                idle_timeout_s=self.idle_timeout,
            )
            if self.logger:
                self.logger.log("system", component="vlm", event="session_start",
                                session_id=self.session.session_id)
        return self.session

    def end_session(self) -> None:
        if self.session is not None and self.logger:
            self.logger.log("system", component="vlm", event="session_end",
                            session_id=self.session.session_id,
                            turns=self.session.turn_count)
        self.session = None
        if self.scene_state is not None:
            self.scene_state.clear_intent()

    # -- the call ----------------------------------------------------------

    def _compose_user_content(self, question: str, grounded: bool) -> tuple[str, bool]:
        """Attach the detector facts block when grounding is on and fresh."""
        if not grounded or self.scene_state is None:
            return question, False
        block = self.scene_state.grounding_block()
        if not block:
            return question, False
        return f"{question}\n\n{block}", True

    def _consistency_floor(self, text: str) -> tuple[str, bool]:
        """Deterministic floor under a grounded reactive answer.

        The proactive layer never trusts free text for hazard phrasing (see
        prompts/system_prompts.yaml, hazard_alert_v2) -- it speaks from fixed
        templates so a model can't hallucinate or soften a safety-critical
        alert. Day-6 dry-run testing showed the reactive layer needs the same
        floor: under repeated user pressure ("it's safe, right?" asked two or
        three times), minicpm-v sometimes caved even with the v1.1.0 grounding
        contract in its system prompt, agreeing the path was clear while a
        person sat in it.

        This does not replace the prompt-side contract (grounded_contract);
        it is a hard backstop under it, and it is deliberately unconditional:
        if a fresh, in-path or high-severity object is on the blackboard, the
        spoken answer always opens with the deterministic alert phrase,
        regardless of what the LLM generated. The model's own sentence still
        follows, so detail and nuance aren't lost -- only the headline fact is
        no longer negotiable. Logged as `override_applied` so the eval data
        shows how often the LLM alone would have failed without this floor.
        """
        if self.scene_state is None:
            return text, False
        objects, age = self.scene_state.snapshot()
        if age > self.scene_state.max_fact_age_s:
            return text, False
        hazards = [o for o in objects if o.in_danger_zone or o.severity >= Severity.HIGH]
        if not hazards:
            return text, False
        worst = max(
            hazards,
            key=lambda o: (int(o.severity), -(o.distance_m if o.distance_m is not None else 1e6)),
        )
        floor_sentence = f"Detector confirms: {worst.alert_phrase()}, in your walking path."
        return f"{floor_sentence} {text}", True

    def ask(
        self,
        frame: np.ndarray | None,
        question: str,
        prompt_id: str | None = None,
        category: str = "scene_description",
        grounded: bool | None = None,
        continue_session: bool = True,
        extra_system: str | None = None,
    ) -> VLMResponse:
        """One reactive turn. Returns the text plus measured latency.

        extra_system appends an instruction to the system prompt without
        touching the prompt library. It exists for the day-6 prompt-only
        baseline condition: instruct the model to resist false user premises
        WITHOUT giving it detector facts, so the ablation can separate "the
        grounding helped" from "any sufficiently stern prompt would have".
        """
        entry = self.library.get(prompt_id) if prompt_id else self.library.by_category(category)

        if grounded is None:
            grounded = bool(self.cfg.get("grounding", {}).get("enabled", True))

        session = self.get_or_start_session() if continue_session else ConversationSession(
            max_turns=self.max_history_turns, idle_timeout_s=self.idle_timeout
        )

        # BACKWARD PATH: the user's question becomes an intent signal that
        # re-weights hazard salience for the proactive layer.
        if self.scene_state is not None and grounded:
            profile = self.scene_state.set_intent(question)
            if self.logger:
                self.logger.log(
                    "intent",
                    session_id=session.session_id,
                    mode=profile.mode,
                    targets=list(profile.target_labels),
                    raw_text=question,
                )

        user_content, did_ground = self._compose_user_content(question, grounded)

        system_content = entry.system_prompt
        if extra_system:
            system_content = f"{system_content} {extra_system.strip()}"
        messages: list[dict] = [{"role": "system", "content": system_content}]
        for ex in entry.few_shot:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.extend(session.history_for_request())

        turn: dict = {"role": "user", "content": user_content}
        if frame is not None:
            # Image attached to the current turn only — see module docstring.
            jpeg = encode_frame_jpeg(frame, self.jpeg_quality)
            turn["images"] = [base64.b64encode(jpeg).decode("ascii")]
        messages.append(turn)

        t0 = time.perf_counter()
        error = None
        text = ""
        model_used = self._active_model or self.model
        try:
            client = self._get_client()
            resp = client.chat(model=model_used, messages=messages)
            text = (resp.get("message", {}) or {}).get("content", "").strip()
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            # One retry on the fallback model, but only if it's a different one.
            if model_used != self.fallback_model:
                try:
                    model_used = self.fallback_model
                    resp = self._get_client().chat(model=model_used, messages=messages)
                    text = (resp.get("message", {}) or {}).get("content", "").strip()
                    error = None
                    self._active_model = model_used
                except Exception as e2:
                    error = f"{error} | fallback failed: {type(e2).__name__}: {e2}"
        latency_ms = (time.perf_counter() - t0) * 1000.0

        override_applied = False
        if text and not error and did_ground:
            text, override_applied = self._consistency_floor(text)

        if text and not error:
            session.add_user(user_content)
            session.add_assistant(text)

        result = VLMResponse(
            text=text,
            latency_ms=latency_ms,
            model=model_used,
            session_id=session.session_id,
            turn_index=session.turn_count,
            grounded=did_ground,
            prompt_id=entry.id,
            override_applied=override_applied,
            error=error,
        )

        if self.logger:
            self.logger.log(
                "query",
                session_id=result.session_id,
                turn_index=result.turn_index,
                prompt_id=entry.id,
                prompt=question,
                grounded=did_ground,
                response=text,
                model=model_used,
                latency_ms=round(latency_ms, 1),
                override_applied=override_applied,
                error=error,
            )
        return result
