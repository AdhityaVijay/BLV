"""HUD layout helpers. Pure logic — no window, no camera."""

import pytest

from pipeline.hud import SEVERITY_COLORS, severity_color, wrap_text
from pipeline.scene_state import Severity


def test_wrap_empty():
    assert wrap_text("", 20) == []


def test_wrap_short_text_is_one_line():
    assert wrap_text("person ahead", 40) == ["person ahead"]


def test_wrap_breaks_on_words():
    lines = wrap_text("the quick brown fox jumps over the lazy dog", 12)
    assert all(len(l) <= 12 for l in lines)
    assert " ".join(lines) == "the quick brown fox jumps over the lazy dog"


def test_wrap_hard_splits_overlong_words():
    """A single long token must not silently overflow the panel edge."""
    lines = wrap_text("supercalifragilisticexpialidocious", 10)
    assert all(len(l) <= 10 for l in lines)
    assert "".join(lines) == "supercalifragilisticexpialidocious"


def test_wrap_mixes_long_word_with_normal_text():
    """A hard-split word's tail keeps behaving like a normal word: the next
    word joins it if it fits. -> ['see', 'aaaaaaaa', 'aaaaaaaa', 'aaaa now']"""
    lines = wrap_text("see aaaaaaaaaaaaaaaaaaaa now", 8)
    assert all(len(l) <= 8 for l in lines)
    assert lines[0] == "see"
    assert lines[-1].endswith("now")
    assert "".join(lines).count("a") == 20


def test_wrap_preserves_all_words():
    text = "chair to your left about two metres away"
    assert " ".join(wrap_text(text, 15)).split() == text.split()


def test_every_severity_has_a_colour():
    for sev in Severity:
        assert sev in SEVERITY_COLORS
        assert len(severity_color(sev)) == 3


def test_severity_colours_are_distinct():
    """If two severities share a colour the display stops conveying urgency."""
    colours = [severity_color(s) for s in Severity]
    assert len(set(colours)) == len(colours)


def test_critical_is_the_reddest():
    """Sanity check on the BGR ordering — red channel is index 2."""
    crit = severity_color(Severity.CRITICAL)
    low = severity_color(Severity.LOW)
    assert crit[2] > low[2]
    assert crit[0] < low[0]
