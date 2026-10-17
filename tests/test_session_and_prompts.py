"""Conversation session handling (day 4) and the prompt library (day 5)."""

import pytest

from pipeline.prompt_library import PromptLibrary
from pipeline.tts import SpeechGate
from pipeline.vlm import ConversationSession


# -- conversation sessions -------------------------------------------------

def test_history_is_capped_to_max_turns():
    """Uncapped history is latency creep: prompt processing grows with every
    exchange, which on an 8 GB card is felt within a few turns."""
    s = ConversationSession(max_turns=3)
    for i in range(10):
        s.add_user(f"q{i}")
        s.add_assistant(f"a{i}")
    assert len(s.messages) == 6           # 3 turns x (user + assistant)
    assert s.messages[-1]["content"] == "a9"
    assert s.turn_count == 10             # count is not truncated, only history


def test_session_expires_after_idle_timeout():
    s = ConversationSession(idle_timeout_s=10.0)
    s.add_user("hello")
    assert not s.is_expired(s.last_activity + 5)
    assert s.is_expired(s.last_activity + 11)


def test_each_session_has_a_distinct_id():
    assert ConversationSession().session_id != ConversationSession().session_id


def test_history_for_request_is_a_copy():
    s = ConversationSession()
    s.add_user("hi")
    out = s.history_for_request()
    out[0]["content"] = "mutated"
    assert s.messages[0]["content"] == "hi"


# -- prompt library --------------------------------------------------------

@pytest.fixture(scope="module")
def lib():
    return PromptLibrary()


def test_three_categories_exist(lib):
    for c in ("scene_description", "object_localization", "hazard_alert"):
        assert c in lib.categories()


def test_shared_fragments_are_spliced_not_left_as_placeholders(lib):
    for pid in lib.ids():
        prompt = lib.get(pid).system_prompt
        assert "{spatial_rule}" not in prompt
        assert "{uncertainty_rule}" not in prompt
        assert "{grounded_contract}" not in prompt


def test_every_category_enforces_the_uncertainty_instruction(lib):
    for pid in lib.ids():
        assert "not sure" in lib.get(pid).system_prompt.lower()


def test_every_category_enforces_left_right_not_clock_face(lib):
    for pid in lib.ids():
        p = lib.get(pid).system_prompt.lower()
        assert "left" in p and "right" in p
        assert "never use clock-face" in p


def test_reactive_categories_carry_the_grounding_contract(lib):
    """The anti-sycophancy clause must be present in the categories that talk
    to the user, or the grounded condition in day 6 is a no-op."""
    for category in ("scene_description", "object_localization"):
        p = lib.by_category(category).system_prompt.lower()
        assert "detector facts" in p
        assert "correct them" in p


def test_every_category_has_few_shot_examples(lib):
    for pid in lib.ids():
        assert len(lib.get(pid).few_shot) >= 2


def test_hazard_templates_cover_every_severity(lib):
    t = lib.by_category("hazard_alert").templates
    for level in ("critical", "high", "medium", "low", "target_found"):
        assert level in t


def test_hazard_templates_format_cleanly(lib):
    t = lib.by_category("hazard_alert").templates
    out = t["critical"].format(label="person", position_phrase="directly ahead",
                               distance_phrase="less than a metre away")
    assert out.startswith("Stop.")
    assert "{" not in out


def test_metadata_present_for_versioning(lib):
    for pid in lib.ids():
        e = lib.get(pid)
        assert e.version and e.purpose
        assert isinstance(e.known_failure_modes, list)


# -- speech gate -----------------------------------------------------------

def test_first_utterance_is_always_allowed():
    assert SpeechGate().allows("person:centre:near", now=0.0) is True


def test_same_phrase_suppressed_within_cooldown():
    g = SpeechGate(cooldown_s=6.0, min_gap_s=1.5)
    g.mark("person:centre:near", now=0.0)
    assert g.allows("person:centre:near", now=3.0) is False
    assert g.allows("person:centre:near", now=7.0) is True


def test_global_gap_spaces_out_different_alerts():
    g = SpeechGate(cooldown_s=6.0, min_gap_s=1.5)
    g.mark("person:centre:near", now=0.0)
    assert g.allows("chair:left:mid", now=0.5) is False
    assert g.allows("chair:left:mid", now=2.0) is True


def test_critical_can_bypass_the_global_gap():
    g = SpeechGate(cooldown_s=6.0, min_gap_s=1.5)
    g.mark("chair:left:mid", now=0.0)
    assert g.allows("person:centre:imminent", now=0.2, bypass_gap=True) is True
