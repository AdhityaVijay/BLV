"""Regression tests for the persistent-hazard silence bug.

THE BUG
-------
Critical alerts were exempt from the cooldown gate entirely, on the reasoning
that a safety warning must never be suppressed. But the detector runs at
roughly 10 Hz, so a person simply standing in frame produced a CRITICAL alert
on every cycle. Each one arrived while the previous was still being spoken,
was ruled an interrupt, called engine.stop(), and killed that utterance
mid-word.

The result was the worst possible failure mode: the system was completely
silent, while its logs showed it alerting ten times a second. And it failed
hardest in the most dangerous situation, because the closer the hazard, the
more reliably it re-triggered.

These tests pin the fix at the level of the rules, so it cannot regress
quietly.
"""

from pipeline.arbitration import Arbiter, Decision, TurnState
from pipeline.scene_state import Severity
from pipeline.tts import SpeechGate

DETECTOR_HZ = 10.0
TICK = 1.0 / DETECTOR_HZ
CRITICAL_COOLDOWN = 2.5


def _gate() -> SpeechGate:
    return SpeechGate(cooldown_s=6.0, min_gap_s=1.5)


def _run_persistent_hazard(seconds: float, severity: Severity) -> int:
    """Simulate one unchanging object seen every detector tick. Returns how
    many alerts would actually be spoken."""
    gate = _gate()
    arbiter = Arbiter(interrupt_threshold=Severity.CRITICAL, queue_threshold=Severity.HIGH)
    key = "person:centre:imminent"
    spoken = 0
    t = 0.0
    while t < seconds:
        is_critical = severity >= Severity.CRITICAL
        allowed = gate.allows(
            key, now=t,
            bypass_gap=is_critical,
            cooldown_override=CRITICAL_COOLDOWN if is_critical else None,
        )
        result = arbiter.decide(severity, TurnState.IDLE, allowed)
        if result.speaks_now:
            spoken += 1
            gate.mark(key, now=t)
        t += TICK
    return spoken


def test_persistent_critical_does_not_fire_every_tick():
    """The bug: 100 alerts in 10 seconds, each killing the last."""
    spoken = _run_persistent_hazard(10.0, Severity.CRITICAL)
    assert spoken < 10, f"critical re-fired {spoken} times in 10s — cooldown not applied"


def test_persistent_critical_still_repeats_while_danger_lasts():
    """But it must not go quiet either. Standing in front of a hazard should
    keep producing warnings, just at a listenable pace."""
    spoken = _run_persistent_hazard(10.0, Severity.CRITICAL)
    assert spoken >= 3, f"critical only spoke {spoken} times in 10s — too quiet"


def test_critical_repeat_rate_matches_its_cooldown():
    spoken = _run_persistent_hazard(10.0, Severity.CRITICAL)
    expected = int(10.0 / CRITICAL_COOLDOWN)
    assert abs(spoken - expected) <= 1, f"expected ~{expected} alerts, got {spoken}"


def test_persistent_high_uses_the_normal_longer_cooldown():
    spoken = _run_persistent_hazard(12.0, Severity.HIGH)
    assert 1 <= spoken <= 3, f"HIGH spoke {spoken} times in 12s"


def test_critical_repeats_more_often_than_high():
    """Urgency should map to repetition rate, not to nothing at all."""
    crit = _run_persistent_hazard(12.0, Severity.CRITICAL)
    high = _run_persistent_hazard(12.0, Severity.HIGH)
    assert crit > high


# -- the gate's two checks are genuinely independent ------------------------

def test_critical_still_bypasses_the_global_gap():
    """A critical hazard must not be delayed because something ELSE spoke a
    moment ago — that part of the original design was right."""
    gate = _gate()
    gate.mark("chair:left:mid", now=0.0)
    assert gate.allows("person:centre:imminent", now=0.2, bypass_gap=True) is True


def test_critical_does_not_bypass_its_own_cooldown():
    gate = _gate()
    gate.mark("person:centre:imminent", now=0.0)
    assert gate.allows("person:centre:imminent", now=0.5,
                       bypass_gap=True, cooldown_override=CRITICAL_COOLDOWN) is False
    assert gate.allows("person:centre:imminent", now=3.0,
                       bypass_gap=True, cooldown_override=CRITICAL_COOLDOWN) is True


def test_a_different_critical_object_is_not_blocked():
    """Cooldown is per-key, so a NEW hazard appearing is announced immediately
    even while another is cooling down."""
    gate = _gate()
    gate.mark("person:centre:imminent", now=0.0)
    assert gate.allows("car:right:imminent", now=0.3,
                       bypass_gap=True, cooldown_override=CRITICAL_COOLDOWN) is True


def test_arbiter_now_honours_cooldown_for_critical():
    """The arbiter's critical exemption is gone; the gate owns this decision."""
    arbiter = Arbiter()
    result = arbiter.decide(Severity.CRITICAL, TurnState.IDLE, gate_allows=False)
    assert result.decision == Decision.SUPPRESSED


def test_escalating_hazard_is_announced_immediately():
    """An object getting closer changes distance band, hence key, so the
    escalation is spoken at once rather than waiting out a cooldown."""
    gate = _gate()
    gate.mark("person:centre:near", now=0.0)
    assert gate.allows("person:centre:imminent", now=0.4,
                       bypass_gap=True, cooldown_override=CRITICAL_COOLDOWN) is True
