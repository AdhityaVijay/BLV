from blv.audio.alert_manager import AlertManager


def test_first_alert_always_speaks():
    mgr = AlertManager(alert_cooldown_seconds=4.0, min_priority_gap_seconds=0.8)
    assert mgr.should_speak("person:left", now=0.0) is True


def test_repeat_within_cooldown_is_suppressed():
    mgr = AlertManager(alert_cooldown_seconds=4.0, min_priority_gap_seconds=0.8)
    mgr.mark_spoken("person:left", now=0.0)
    assert mgr.should_speak("person:left", now=1.0) is False


def test_repeat_after_cooldown_speaks_again():
    mgr = AlertManager(alert_cooldown_seconds=4.0, min_priority_gap_seconds=0.8)
    mgr.mark_spoken("person:left", now=0.0)
    assert mgr.should_speak("person:left", now=5.0) is True


def test_different_key_blocked_by_global_gap():
    mgr = AlertManager(alert_cooldown_seconds=4.0, min_priority_gap_seconds=0.8)
    mgr.mark_spoken("person:left", now=0.0)
    # different key, but too soon after ANY alert
    assert mgr.should_speak("car:right", now=0.3) is False


def test_different_key_after_gap_speaks():
    mgr = AlertManager(alert_cooldown_seconds=4.0, min_priority_gap_seconds=0.8)
    mgr.mark_spoken("person:left", now=0.0)
    assert mgr.should_speak("car:right", now=1.0) is True
