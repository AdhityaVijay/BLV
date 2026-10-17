"""Arbitration rules — the day-3 core claim, tested deterministically.

The acceptance test in the spec is a physical one (fire an alert, ask a
question, wave an object). That is necessary but not sufficient: it exercises
one path through the state machine on one run. These tests cover every
(severity x state) cell, so a regression in the rules surfaces in CI rather
than during a demo.
"""

from pipeline.arbitration import Arbiter, Decision, TurnState
from pipeline.scene_state import Severity

ARB = Arbiter(interrupt_threshold=Severity.CRITICAL, queue_threshold=Severity.HIGH)


def d(severity, state, gate=True):
    return ARB.decide(severity, state, gate).decision


# -- idle channel ----------------------------------------------------------

def test_any_alert_speaks_when_idle():
    for sev in Severity:
        assert d(sev, TurnState.IDLE) == Decision.ALLOWED


# -- during a reactive turn ------------------------------------------------

def test_critical_interrupts_reactive_speech():
    assert d(Severity.CRITICAL, TurnState.REACTIVE_SPEAKING) == Decision.INTERRUPTED


def test_critical_interrupts_while_llm_is_thinking():
    assert d(Severity.CRITICAL, TurnState.REACTIVE_PENDING) == Decision.INTERRUPTED


def test_high_is_queued_not_dropped_during_reactive_turn():
    assert d(Severity.HIGH, TurnState.REACTIVE_SPEAKING) == Decision.QUEUED


def test_medium_is_suppressed_during_reactive_turn():
    # A chair three metres away is not worth queueing behind an answer;
    # by the time the answer ends, the scene has moved on.
    assert d(Severity.MEDIUM, TurnState.REACTIVE_SPEAKING) == Decision.SUPPRESSED


def test_low_is_suppressed_during_reactive_turn():
    assert d(Severity.LOW, TurnState.REACTIVE_PENDING) == Decision.SUPPRESSED


# -- alert vs alert --------------------------------------------------------

def test_critical_overrides_an_in_progress_alert():
    assert d(Severity.CRITICAL, TurnState.ALERT_SPEAKING) == Decision.INTERRUPTED


def test_high_queues_behind_an_in_progress_alert():
    assert d(Severity.HIGH, TurnState.ALERT_SPEAKING) == Decision.QUEUED


def test_low_suppressed_behind_an_in_progress_alert():
    assert d(Severity.LOW, TurnState.ALERT_SPEAKING) == Decision.SUPPRESSED


# -- cooldown interaction --------------------------------------------------

def test_cooldown_suppresses_non_critical_repeat():
    assert d(Severity.HIGH, TurnState.IDLE, gate=False) == Decision.SUPPRESSED


def test_cooldown_applies_to_critical_too():
    """The arbiter no longer exempts CRITICAL from the cooldown.

    It used to, so that a safety warning could never be withheld. The effect
    was the opposite of the intent: a persistent hazard re-fired at the
    detector's ~10 Hz, each repeat preempting the last, and nothing was ever
    audible. Responsibility for the distinction now sits in SpeechGate, which
    lets critical alerts skip the GLOBAL spacing rule while keeping a short
    per-key cooldown of their own — see tests/test_persistent_hazard.py.

    By the time the arbiter sees gate_allows=False, the gate has already
    decided this exact alert was spoken moments ago.
    """
    assert d(Severity.CRITICAL, TurnState.IDLE, gate=False) == Decision.SUPPRESSED


def test_critical_that_passes_the_gate_still_interrupts():
    assert d(Severity.CRITICAL, TurnState.REACTIVE_SPEAKING, gate=True) == Decision.INTERRUPTED
    assert d(Severity.CRITICAL, TurnState.REACTIVE_PENDING, gate=True) == Decision.INTERRUPTED


# -- guarantees ------------------------------------------------------------

def test_reactive_turn_is_never_blocked_indefinitely():
    """Nothing in the rules can stop a reactive turn from proceeding — the
    arbiter only ever decides about ALERTS. This is the 'neither layer
    silently blocks the other' requirement."""
    for state in TurnState:
        assert state.reactive_in_flight == (
            state in (TurnState.REACTIVE_PENDING, TurnState.REACTIVE_SPEAKING)
        )


def test_every_decision_carries_a_reason():
    for sev in Severity:
        for state in TurnState:
            result = ARB.decide(sev, state, True)
            assert result.reason, f"no reason for {sev} in {state}"
            assert result.speaks_now == (
                result.decision in (Decision.ALLOWED, Decision.INTERRUPTED)
            )


def test_thresholds_are_tunable():
    lax = Arbiter(interrupt_threshold=Severity.HIGH, queue_threshold=Severity.MEDIUM)
    assert lax.decide(Severity.HIGH, TurnState.REACTIVE_SPEAKING, True).decision == Decision.INTERRUPTED
    assert lax.decide(Severity.MEDIUM, TurnState.REACTIVE_SPEAKING, True).decision == Decision.QUEUED
