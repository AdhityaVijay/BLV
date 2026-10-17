"""The novel component: grounded scene-state blackboard.

Covers both directions of the loop — detector facts flowing into prompts, and
user intent flowing back into hazard salience — plus the ablation switch that
day 6 depends on.
"""

import time

import pytest

from pipeline.scene_state import (
    IntentMode,
    SceneState,
    Severity,
    TrackedObject,
    parse_intent,
)


def make_obj(label="person", position="centre", distance=1.2, severity=Severity.HIGH,
             in_zone=True, conf=0.91):
    return TrackedObject(
        label=label, confidence=conf, bbox=(100, 100, 200, 400),
        position=position, distance_m=distance,
        distance_band="near" if distance and distance < 2 else "mid",
        area_fraction=0.2, in_danger_zone=in_zone, severity=severity,
    )


# -- forward path: detector -> VLM grounding -------------------------------

def test_grounding_block_contains_measurements():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("person", "centre", 1.2)])
    block = s.grounding_block()
    assert "DETECTOR FACTS" in block
    assert "person" in block
    assert "1.2 m" in block
    assert "centre" in block


def test_grounding_block_instructs_the_model_to_correct_the_user():
    """This sentence is the anti-sycophancy mechanism. If it disappears, the
    day-6 ablation stops measuring anything."""
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj()])
    assert "correct the user" in s.grounding_block().lower()


def test_grounding_disabled_returns_empty_block():
    """The day-6 ablation switch."""
    s = SceneState(grounding_enabled=False)
    s.update_objects([make_obj()])
    assert s.grounding_block() == ""


def test_stale_facts_are_not_sent_as_ground_truth():
    """Sending two-second-old geometry as fact would trade sycophancy for
    confidently wrong grounding."""
    s = SceneState(grounding_enabled=True, max_fact_age_s=0.05)
    s.update_objects([make_obj()])
    time.sleep(0.1)
    assert s.grounding_block() == ""


def test_empty_detection_does_not_claim_the_space_is_safe():
    """An empty detector result means 'no KNOWN classes found', not 'clear'.
    The distinction is the difference between a hedge and a false all-clear."""
    s = SceneState(grounding_enabled=True)
    s.update_objects([])
    block = s.grounding_block().lower()
    assert "no recognisable objects" in block
    assert "does not prove" in block or "not prove" in block


def test_grounding_block_is_severity_ordered_and_capped():
    s = SceneState(grounding_enabled=True, max_objects_in_prompt=2)
    s.update_objects([
        make_obj("chair", "left", 3.0, Severity.LOW),
        make_obj("person", "centre", 0.8, Severity.CRITICAL),
        make_obj("dog", "right", 2.0, Severity.MEDIUM),
    ])
    lines = [l for l in s.grounding_block().splitlines() if l.startswith("- ")]
    assert len(lines) == 2
    assert "person" in lines[0]


# -- intent parsing --------------------------------------------------------

def test_navigation_intent():
    assert parse_intent("can I walk forward?").mode == IntentMode.NAVIGATING
    assert parse_intent("is my path clear").mode == IntentMode.NAVIGATING


def test_search_intent_with_target():
    p = parse_intent("where is my cup", known_labels={"cup", "chair"})
    assert p.mode == IntentMode.SEARCHING
    assert "cup" in p.target_labels


def test_search_without_known_target_still_searches():
    assert parse_intent("find my glasses").mode == IntentMode.SEARCHING


def test_descriptive_question_is_inspecting():
    assert parse_intent("what's on the table?").mode == IntentMode.INSPECTING


def test_empty_text_is_unknown():
    assert parse_intent("").mode == IntentMode.UNKNOWN


# -- backward path: intent -> salience -------------------------------------

def test_navigating_boosts_path_obstacles():
    s = SceneState(grounding_enabled=True, intent_navigating_boost=1)
    s.update_objects([make_obj()])
    s.set_intent("can I walk forward?")
    obj = make_obj(severity=Severity.MEDIUM, in_zone=True)
    assert s.apply_salience(obj) == Severity.HIGH


def test_navigating_does_not_boost_objects_outside_the_path():
    s = SceneState(grounding_enabled=True)
    s.set_intent("can I walk forward?")
    obj = make_obj(severity=Severity.MEDIUM, in_zone=False)
    assert s.apply_salience(obj) == Severity.MEDIUM


def test_inspecting_damps_chatter():
    s = SceneState(grounding_enabled=True, intent_inspecting_damp=1)
    s.set_intent("what's on the table?")
    obj = make_obj(severity=Severity.MEDIUM)
    assert s.apply_salience(obj) == Severity.LOW


def test_inspecting_never_damps_critical():
    """Safety is not negotiable: a critical hazard is never quietened because
    the user happens to be listening to a description."""
    s = SceneState(grounding_enabled=True)
    s.set_intent("describe the room")
    obj = make_obj(severity=Severity.CRITICAL)
    assert s.apply_salience(obj) == Severity.CRITICAL


def test_salience_is_identity_when_grounding_disabled():
    s = SceneState(grounding_enabled=False)
    s.set_intent("can I walk forward?")
    obj = make_obj(severity=Severity.MEDIUM, in_zone=True)
    assert s.apply_salience(obj) == Severity.MEDIUM


# -- backward path: proactive answer to a search ---------------------------

def test_target_announced_when_searched_object_appears():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("cup", "right", 0.5, Severity.LOW, in_zone=False)])
    s.set_intent("where is my cup")
    found = s.pending_target_announcement()
    assert found is not None and found.label == "cup"


def test_target_not_reannounced_within_cooldown():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("cup", "right", 0.5, Severity.LOW, in_zone=False)])
    s.set_intent("where is my cup")
    assert s.pending_target_announcement(cooldown_s=60) is not None
    assert s.pending_target_announcement(cooldown_s=60) is None


def test_no_target_announcement_without_search_intent():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("cup", "right", 0.5)])
    s.set_intent("what's in front of me")
    assert s.pending_target_announcement() is None


def test_no_target_announcement_when_grounding_disabled():
    s = SceneState(grounding_enabled=False)
    s.update_objects([make_obj("cup")])
    s.set_intent("where is my cup")
    assert s.pending_target_announcement() is None


# -- phrasing convention ---------------------------------------------------

def test_alert_phrase_uses_left_right_distance_not_clock_face():
    obj = make_obj("chair", "left", 1.1)
    phrase = obj.alert_phrase()
    assert "to your left" in phrase
    assert "o'clock" not in phrase
    assert "metre" in phrase


def test_centre_objects_are_described_as_directly_ahead():
    assert "directly ahead" in make_obj("person", "centre", 0.9).alert_phrase()


def test_dedup_key_changes_when_object_gets_closer():
    """Escalation must be announceable: the same object moving from 'near' to
    'imminent' is new information, not a repeat."""
    far = make_obj("person", "centre", 1.8)
    near = make_obj("person", "centre", 0.6)
    near.distance_band = "imminent"
    assert far.key != near.key
