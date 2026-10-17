"""New in v1.2.0: VLM._consistency_floor.

Dry-run testing on 2026-10-17 showed minicpm-v caving under repeated user
pressure even with the v1.1.0 grounding contract in its system prompt -- it
agreed a path was clear while a person sat in it. This is a deterministic,
code-level backstop under the prompt-side contract: when a fresh in-path or
high-severity object is on the blackboard, the spoken answer always opens
with the detector's own fact, regardless of what the LLM generated.

No Ollama/cv2/torch needed -- _consistency_floor only reads the scene_state
blackboard, so this is pure logic and runs anywhere.
"""

from pipeline.scene_state import SceneState, Severity, TrackedObject
from pipeline.vlm import VLM


def make_obj(label="person", position="centre", distance=1.2, severity=Severity.HIGH,
             in_zone=True, conf=0.91):
    return TrackedObject(
        label=label, confidence=conf, bbox=(100, 100, 200, 400),
        position=position, distance_m=distance,
        distance_band="near" if distance and distance < 2 else "mid",
        area_fraction=0.2, in_danger_zone=in_zone, severity=severity,
    )


def make_vlm(scene_state):
    cfg = {"vlm": {"model": "minicpm-v", "fallback_model": "llama3.2-vision"}}
    return VLM(cfg, scene_state=scene_state)


def test_floor_prefixes_the_deterministic_fact_when_hazard_is_fresh():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("person", "centre", 2.9, severity=Severity.HIGH, in_zone=True)])
    vlm = make_vlm(s)

    text, applied = vlm._consistency_floor("Yes, it's safe to walk forward.")

    assert applied is True
    assert text.startswith("Detector confirms:")
    assert "person" in text
    assert "Yes, it's safe to walk forward." in text   # model's own text is kept, not discarded


def test_floor_does_nothing_when_no_hazard_present():
    s = SceneState(grounding_enabled=True)
    s.update_objects([])  # empty room, nothing to override with
    vlm = make_vlm(s)

    text, applied = vlm._consistency_floor("Nothing visible from here.")

    assert applied is False
    assert text == "Nothing visible from here."


def test_floor_does_nothing_when_detector_facts_are_stale():
    s = SceneState(grounding_enabled=True, max_fact_age_s=2.0)
    s.update_objects([make_obj("person", "centre", 1.0, severity=Severity.CRITICAL, in_zone=True)])
    s._updated_at -= 10.0  # simulate a 10s-old observation, past max_fact_age_s
    vlm = make_vlm(s)

    text, applied = vlm._consistency_floor("Path looks clear.")

    assert applied is False
    assert text == "Path looks clear."


def test_floor_applies_even_when_model_already_named_the_hazard():
    """The exact failure seen in the dry run: the model named the person but
    still called the path clear/safe. The floor is unconditional on content,
    so it fires here too rather than trying to parse whether the model's
    phrasing was "good enough"."""
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj("person", "centre", 3.1, severity=Severity.HIGH, in_zone=True)])
    vlm = make_vlm(s)

    model_text = (
        "It appears safe to walk forward. There is no visible obstacle directly "
        "in front of you. However, be cautious of the person sitting in front of "
        "you, as you are close enough to be detected as a potential obstacle."
    )
    text, applied = vlm._consistency_floor(model_text)

    assert applied is True
    assert text.startswith("Detector confirms:")


def test_floor_picks_the_most_severe_object_when_several_are_present():
    s = SceneState(grounding_enabled=True)
    s.update_objects([
        make_obj("chair", "left", 4.0, severity=Severity.MEDIUM, in_zone=False),
        make_obj("person", "centre", 1.0, severity=Severity.CRITICAL, in_zone=True),
    ])
    vlm = make_vlm(s)

    text, applied = vlm._consistency_floor("All clear ahead.")

    assert applied is True
    assert "person" in text.split(".")[0]   # the critical hazard, not the chair, leads
