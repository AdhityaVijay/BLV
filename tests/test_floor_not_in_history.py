"""Regression test for a failure found during real dry-run testing on
2026-10-17: the floor sentence from VLM._consistency_floor was being written
into ConversationSession history, and on the next turn minicpm-v started
imitating the literal "Detector confirms:" phrasing to dress up its OWN
contradicting claims with false authority (e.g. "Detector confirms: it's
safe to walk forward ... according to the detector" -- fabricated, never
produced by the real detector). The fix: history stores the model's raw
text; the floor is applied only to the text returned/spoken for that turn.
"""

from unittest.mock import MagicMock, patch

from pipeline.scene_state import SceneState, Severity, TrackedObject
from pipeline.vlm import VLM


def make_obj(label="person", position="centre", distance=3.0, severity=Severity.HIGH,
             in_zone=True, conf=0.9):
    return TrackedObject(
        label=label, confidence=conf, bbox=(100, 100, 200, 400),
        position=position, distance_m=distance,
        distance_band="mid", area_fraction=0.2, in_danger_zone=in_zone, severity=severity,
    )


def test_floor_sentence_never_enters_conversation_history():
    s = SceneState(grounding_enabled=True)
    s.update_objects([make_obj()])

    cfg = {"vlm": {"model": "minicpm-v", "fallback_model": "llama3.2-vision"}}
    vlm = VLM(cfg, scene_state=s)

    fake_client = MagicMock()
    fake_client.chat.return_value = {
        "message": {"content": "The walking path is clear directly ahead of you."}
    }
    with patch.object(VLM, "_get_client", return_value=fake_client):
        result = vlm.ask(frame=None, question="is it safe to walk forward?")

    # The person listening on THIS turn gets the floor fact up front.
    assert result.override_applied is True
    assert result.text.startswith("Detector confirms:")

    # But the conversation history the model will see on the NEXT turn must
    # hold only what it actually said -- never the floor sentence, so it has
    # nothing to imitate.
    history = vlm.session.history_for_request()
    assistant_turns = [m["content"] for m in history if m["role"] == "assistant"]
    assert assistant_turns == ["The walking path is clear directly ahead of you."]
    assert all("Detector confirms" not in t for t in assistant_turns)
