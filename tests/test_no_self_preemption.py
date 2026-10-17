"""The 'do not preempt an equally urgent alert' rule, now inside the arbiter.

This lived in the orchestrator as a check applied AFTER the arbiter had
already returned INTERRUPTED and after that decision had been logged. Two
consequences, both bad: the session log recorded 151 interrupts that never
happened, and the rule itself was untestable without standing up threads, a
camera and a speech engine.
"""

from pipeline.arbitration import Arbiter, Decision, TurnState
from pipeline.scene_state import Severity

ARB = Arbiter(interrupt_threshold=Severity.CRITICAL, queue_threshold=Severity.HIGH)


def test_critical_does_not_preempt_another_critical():
    r = ARB.decide(Severity.CRITICAL, TurnState.ALERT_SPEAKING, True,
                   speaking_severity=Severity.CRITICAL)
    assert r.decision == Decision.SUPPRESSED
    assert "already speaking" in r.reason


def test_critical_does_preempt_a_lesser_alert():
    r = ARB.decide(Severity.CRITICAL, TurnState.ALERT_SPEAKING, True,
                   speaking_severity=Severity.MEDIUM)
    assert r.decision == Decision.INTERRUPTED


def test_lesser_alert_never_preempts_a_critical():
    r = ARB.decide(Severity.HIGH, TurnState.ALERT_SPEAKING, True,
                   speaking_severity=Severity.CRITICAL)
    assert r.decision == Decision.SUPPRESSED


def test_equal_high_alerts_do_not_fight():
    r = ARB.decide(Severity.HIGH, TurnState.ALERT_SPEAKING, True,
                   speaking_severity=Severity.HIGH)
    assert r.decision == Decision.SUPPRESSED


def test_unknown_speaking_severity_falls_back_to_old_behaviour():
    """When the caller doesn't know what's speaking, a critical still cuts in
    — failing toward safety rather than toward silence."""
    r = ARB.decide(Severity.CRITICAL, TurnState.ALERT_SPEAKING, True,
                   speaking_severity=None)
    assert r.decision == Decision.INTERRUPTED


def test_speaking_severity_is_irrelevant_during_a_reactive_turn():
    """A reactive answer is not an alert, so the no-preemption rule must not
    accidentally protect it from a genuine safety interrupt."""
    r = ARB.decide(Severity.CRITICAL, TurnState.REACTIVE_SPEAKING, True,
                   speaking_severity=Severity.CRITICAL)
    assert r.decision == Decision.INTERRUPTED


def test_decision_is_final_no_post_hoc_override_needed():
    """Every (severity, speaking_severity) combination resolves inside decide(),
    so what gets logged is what actually happens."""
    for sev in Severity:
        for speaking in list(Severity) + [None]:
            r = ARB.decide(sev, TurnState.ALERT_SPEAKING, True, speaking_severity=speaking)
            assert r.reason
            if r.decision == Decision.INTERRUPTED:
                assert speaking is None or sev > speaking
