"""McNemar / binomial tests, checked against hand-computed values.

Getting a p-value wrong is worse than having no p-value, because it looks
authoritative. Each expected number here is computed by hand in the comment.
"""

import pytest

from pipeline.stats import (
    exact_binomial_two_sided,
    mcnemar_exact,
    pair_outcomes,
)


# -- binomial --------------------------------------------------------------

def test_binomial_n_zero_is_one():
    assert exact_binomial_two_sided(0, 0) == 1.0


def test_binomial_all_one_way():
    # n=10, k=0: 2 * C(10,0) * 0.5^10 = 2/1024 = 0.001953125
    assert exact_binomial_two_sided(0, 10) == pytest.approx(0.001953125)


def test_binomial_even_split_is_not_significant():
    # n=10, k=5: 2 * sum(C(10,0..5))/1024 = 2*638/1024 > 1 -> capped at 1.0
    assert exact_binomial_two_sided(5, 10) == 1.0


def test_binomial_lopsided():
    # n=10, k=2: 2 * (C(10,0)+C(10,1)+C(10,2))/1024 = 2*56/1024 = 0.109375
    assert exact_binomial_two_sided(2, 10) == pytest.approx(0.109375)


def test_binomial_is_symmetric():
    assert exact_binomial_two_sided(3, 10) == exact_binomial_two_sided(7, 10)


def test_binomial_rejects_out_of_range():
    with pytest.raises(ValueError):
        exact_binomial_two_sided(11, 10)


# -- McNemar ---------------------------------------------------------------

def test_mcnemar_concordant_pairs_are_ignored():
    """Pairs where both conditions agree carry no information about which is
    better, and must not influence the p-value."""
    a = [True, True, False, False]
    b = [True, True, False, False]
    r = mcnemar_exact(a, b)
    assert r.n_discordant == 0
    assert r.p_value == 1.0
    assert r.concordant_pass == 2
    assert r.concordant_fail == 2


def test_mcnemar_perfect_improvement():
    # 10 pairs, ungrounded fails all, grounded passes all -> b=0, c=10
    ungrounded = [False] * 10
    grounded = [True] * 10
    r = mcnemar_exact(ungrounded, grounded)
    assert r.b == 0 and r.c == 10
    assert r.p_value == pytest.approx(0.001953125)
    assert "significant" in r.verdict()
    assert "NOT significant" not in r.verdict()


def test_mcnemar_detects_direction():
    ungrounded = [False] * 8 + [True] * 2
    grounded = [True] * 8 + [False] * 2
    r = mcnemar_exact(ungrounded, grounded)
    assert r.c == 8 and r.b == 2
    assert "B better than A" in r.verdict()


def test_mcnemar_small_effect_is_not_significant():
    """Guards against over-claiming: 6-vs-4 on ten disagreements is noise, and
    the test must say so rather than rewarding a tempting-looking gap."""
    a = [True] * 6 + [False] * 4
    b = [False] * 6 + [True] * 4
    r = mcnemar_exact(a, b)
    assert r.n_discordant == 10
    assert r.p_value > 0.05
    assert "NOT significant" in r.verdict()


def test_mcnemar_counts_add_up():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    r = mcnemar_exact(a, b)
    assert r.n_pairs == 5
    assert r.b + r.c + r.concordant_pass + r.concordant_fail == 5


def test_mcnemar_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        mcnemar_exact([True, False], [True])


# -- pairing ---------------------------------------------------------------

def test_pair_outcomes_aligns_on_shared_keys():
    a = {("syc_01", "desk.jpg"): True, ("syc_02", "desk.jpg"): False}
    b = {("syc_01", "desk.jpg"): False, ("syc_02", "desk.jpg"): True}
    xs, ys, dropped = pair_outcomes(a, b)
    assert xs == [True, False]
    assert ys == [False, True]
    assert dropped == []


def test_pair_outcomes_drops_unmatched_and_reports_them():
    """An unpaired observation can't contribute to a paired test. Dropping it
    silently would hide missing data; counting it as a failure would invent
    signal that isn't there."""
    a = {("syc_01", "desk.jpg"): True, ("syc_02", "desk.jpg"): True}
    b = {("syc_01", "desk.jpg"): False}
    xs, ys, dropped = pair_outcomes(a, b)
    assert len(xs) == len(ys) == 1
    assert ("syc_02", "desk.jpg") in dropped
