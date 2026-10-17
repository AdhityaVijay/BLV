"""Turn-taking arbitration — the rules, separated from the threading.

This module is pure logic on purpose. The orchestrator owns threads, camera
and audio; this owns the decision "given what is currently being said, what
should happen to this alert?". Keeping them apart is what makes the core
claim of the project testable: the arbitration rules are unit tested
deterministically (tests/test_arbitration.py) instead of only being observable
by waving a chair at a webcam.

STATE MACHINE
-------------
    IDLE               nothing in flight
    REACTIVE_PENDING   a question is with the LLM, no audio yet
    REACTIVE_SPEAKING  the LLM's answer is being read aloud
    ALERT_SPEAKING     a proactive alert is being read aloud

DECISIONS
---------
    allowed      speak it now
    interrupted  cut off current speech and speak it now  (CRITICAL only)
    queued       hold it until the reactive turn finishes
    suppressed   drop it (too low priority to spend the audio channel on,
                 or cooling down)

Every decision is logged with its reason — that log is the evidence for how
the two layers actually interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pipeline.scene_state import Severity


class TurnState(Enum):
    IDLE = "idle"
    REACTIVE_PENDING = "reactive_pending"
    REACTIVE_SPEAKING = "reactive_speaking"
    ALERT_SPEAKING = "alert_speaking"

    @property
    def reactive_in_flight(self) -> bool:
        return self in (TurnState.REACTIVE_PENDING, TurnState.REACTIVE_SPEAKING)


class Decision(str, Enum):
    ALLOWED = "allowed"
    INTERRUPTED = "interrupted"
    QUEUED = "queued"
    SUPPRESSED = "suppressed"


@dataclass
class ArbitrationResult:
    decision: Decision
    reason: str

    @property
    def speaks_now(self) -> bool:
        return self.decision in (Decision.ALLOWED, Decision.INTERRUPTED)


@dataclass
class Arbiter:
    """Rules for whether a proactive alert may use the audio channel.

    interrupt_threshold: severity at or above which an alert cuts off an
        in-progress reactive answer. Default CRITICAL — safety beats
        conversation, but only genuine safety, or the assistant becomes
        impossible to hold a conversation with.
    queue_threshold: severity at or above which an alert is held for after the
        reactive turn rather than dropped. Below it, the alert is suppressed:
        during a conversation, a chair four metres away is not worth queueing,
        because by the time the answer finishes the scene has moved on.
    """

    interrupt_threshold: Severity = Severity.CRITICAL
    queue_threshold: Severity = Severity.HIGH

    def decide(
        self,
        severity: Severity,
        state: TurnState,
        gate_allows: bool,
        speaking_severity: Severity | None = None,
    ) -> ArbitrationResult:
        """speaking_severity is the severity of the alert currently being
        spoken, when one is. It belongs here rather than in the caller: the
        rule "do not preempt something at least as urgent as you" is an
        arbitration rule, and having it applied after this function returned
        meant the logged decision was not the decision that was carried out.
        """
        # Cooldown applies to EVERY severity, critical included.
        #
        # Critical alerts used to be exempt here, on the reasoning that a
        # safety warning must never be withheld. In practice that made a
        # standing person re-fire at detection rate, with each repeat cutting
        # off the last, so the critical case was the one you could never hear.
        # The gate itself now distinguishes the two concerns: critical hazards
        # skip the global spacing rule but keep a short per-key cooldown, so
        # they still repeat while the danger persists - just at a pace a
        # listener can follow.
        if not gate_allows:
            return ArbitrationResult(Decision.SUPPRESSED, "cooldown: alert repeated too soon")

        if not state.reactive_in_flight and state != TurnState.ALERT_SPEAKING:
            return ArbitrationResult(Decision.ALLOWED, "channel idle")

        if state == TurnState.ALERT_SPEAKING:
            # Another alert is talking. Only a strictly MORE urgent one cuts in.
            #
            # Without this, two criticals in succession kill each other
            # mid-word, and the more dangerous the scene the less the user
            # actually hears - which is how a persistent hazard produced a
            # hundred logged interrupts and total silence.
            if speaking_severity is not None and severity <= speaking_severity:
                return ArbitrationResult(
                    Decision.SUPPRESSED,
                    f"{speaking_severity.name} alert already speaking; "
                    f"not preempting with an equally urgent one",
                )
            if severity >= self.interrupt_threshold:
                return ArbitrationResult(
                    Decision.INTERRUPTED, f"{severity.name} alert overrides in-progress alert"
                )
            if severity >= self.queue_threshold:
                return ArbitrationResult(Decision.QUEUED, "alert channel busy, queued behind alert")
            return ArbitrationResult(Decision.SUPPRESSED, "alert channel busy, severity below queue threshold")

        # A reactive turn is in flight (LLM thinking, or answer being spoken).
        if severity >= self.interrupt_threshold:
            return ArbitrationResult(
                Decision.INTERRUPTED,
                f"{severity.name} hazard interrupts reactive turn ({state.value})",
            )
        if severity >= self.queue_threshold:
            return ArbitrationResult(
                Decision.QUEUED, f"reactive turn in flight ({state.value}), queued"
            )
        return ArbitrationResult(
            Decision.SUPPRESSED,
            f"reactive turn in flight ({state.value}), severity below queue threshold",
        )
