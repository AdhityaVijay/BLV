"""Distance estimation and the hazard-severity rule.

These are the tests that matter most: the severity rule decides whether the
system interrupts you, and the distance estimate feeds it. Both run without a
GPU, camera or model.
"""

import math

import pytest

from pipeline.detector import (
    classify_severity,
    distance_band,
    estimate_distance_m,
    focal_length_px,
    horizontal_position,
)
from pipeline.scene_state import Severity

BANDS = {"imminent": 1.0, "near": 2.0, "mid": 4.0}
AREA_ALERT = 0.18
HIGH_RISK = {"person", "car", "bicycle", "dog"}


def test_focal_length_from_fov():
    # 60 deg FOV on a 640px-wide frame -> f = 320 / tan(30 deg) ~= 554 px
    f = focal_length_px(640, 60.0)
    assert f == pytest.approx(320 / math.tan(math.radians(30)), rel=1e-6)
    assert 550 < f < 560


def test_wider_fov_gives_shorter_focal_length():
    assert focal_length_px(640, 90.0) < focal_length_px(640, 60.0)


def test_person_distance_is_plausible():
    # A 1.7 m person filling 340 px of a 480 px frame, f=554 -> ~2.8 m
    f = focal_length_px(640, 60.0)
    d = estimate_distance_m("person", 340, f)
    assert d is not None
    assert 2.0 < d < 3.5


def test_distance_scales_inversely_with_box_height():
    f = focal_length_px(640, 60.0)
    near = estimate_distance_m("person", 400, f)
    far = estimate_distance_m("person", 100, f)
    assert far > near
    assert far == pytest.approx(near * 4, rel=1e-6)


def test_unknown_class_returns_none_rather_than_guessing():
    f = focal_length_px(640, 60.0)
    assert estimate_distance_m("unicorn", 200, f) is None


def test_the_bug_the_naive_heuristic_had():
    """Regression guard for the defect the bbox-height heuristic had.

    Two objects with the SAME box height (300 px) are at completely different
    real distances, because they are different real sizes. The old heuristic
    saw one number — box height — and therefore called them equally close,
    which is how a bus half a room away outranked a chair you were about to
    walk into. The pinhole estimate separates them by their true size ratio.
    """
    f = focal_length_px(640, 60.0)
    bus = estimate_distance_m("bus", 300, f)      # 3.0 m tall -> ~5.5 m away
    chair = estimate_distance_m("chair", 300, f)  # 0.9 m tall -> ~1.7 m away

    # Same pixels, very different distances — the whole point.
    assert bus / chair == pytest.approx(3.0 / 0.9, rel=1e-6)
    assert chair == pytest.approx(1.66, abs=0.05)
    assert bus == pytest.approx(5.54, abs=0.05)

    # And they land in different bands, so they are phrased and prioritised
    # differently rather than collapsing to one "very close" verdict.
    assert distance_band(chair, 0.3, BANDS, AREA_ALERT) == "near"
    assert distance_band(bus, 0.3, BANDS, AREA_ALERT) == "far"


def test_closer_box_of_same_class_escalates_band():
    """Sanity check that the band boundaries are reachable: the same chair
    filling more of the frame does read as imminent."""
    f = focal_length_px(640, 60.0)
    close_chair = estimate_distance_m("chair", 520, f)   # ~0.96 m
    assert distance_band(close_chair, 0.4, BANDS, AREA_ALERT) == "imminent"


def test_distance_bands():
    assert distance_band(0.5, 0.0, BANDS, AREA_ALERT) == "imminent"
    assert distance_band(1.5, 0.0, BANDS, AREA_ALERT) == "near"
    assert distance_band(3.0, 0.0, BANDS, AREA_ALERT) == "mid"
    assert distance_band(9.0, 0.0, BANDS, AREA_ALERT) == "far"


def test_unknown_distance_falls_back_to_area_rule():
    assert distance_band(None, 0.25, BANDS, AREA_ALERT) == "imminent"
    assert distance_band(None, 0.05, BANDS, AREA_ALERT) == "unknown"


def test_horizontal_position_thirds():
    assert horizontal_position(50, 640) == "left"
    assert horizontal_position(320, 640) == "centre"
    assert horizontal_position(600, 640) == "right"


# -- severity rule ---------------------------------------------------------

def _sev(band, in_zone, label="chair", area=0.05):
    return classify_severity(
        band=band, in_danger_zone=in_zone, area_fraction=area,
        label=label, high_risk_classes=HIGH_RISK, area_alert=AREA_ALERT,
    )


def test_imminent_in_path_is_critical():
    assert _sev("imminent", True) == Severity.CRITICAL


def test_imminent_off_path_is_high_not_critical():
    # Something very close but off to the side should not interrupt speech.
    assert _sev("imminent", False) == Severity.HIGH


def test_near_in_path_is_high():
    assert _sev("near", True) == Severity.HIGH


def test_far_off_path_is_low():
    assert _sev("far", False) == Severity.LOW


def test_high_risk_class_escalates_in_path():
    chair = _sev("mid", True, label="chair")
    person = _sev("mid", True, label="person")
    assert person > chair


def test_high_risk_escalation_cannot_exceed_critical():
    assert _sev("imminent", True, label="person") == Severity.CRITICAL


def test_large_box_triggers_high_even_without_distance():
    assert _sev("unknown", False, area=0.4) == Severity.HIGH


def test_severity_bump_clamps():
    assert Severity.CRITICAL.bump(5) == Severity.CRITICAL
    assert Severity.LOW.bump(-5) == Severity.LOW
