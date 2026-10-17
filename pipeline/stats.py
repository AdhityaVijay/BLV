"""Statistics for the paired ablation. No scipy dependency — the exact tests
needed here are a few lines of `math.comb`, and adding a scientific stack to a
real-time pipeline for two functions is not a trade worth making.

WHY McNEMAR AND NOT A CHI-SQUARE OR A t-TEST
--------------------------------------------
The red-team design is PAIRED: the same probe, on the same image, is run under
each condition. So the two samples are not independent, and a test that assumes
independence (chi-square on a 2x2 of totals, or a two-sample t-test) would be
wrong — it throws away the pairing and generally overstates significance.

McNemar's test looks only at the pairs where the conditions DISAGREE:

                          grounded PASS   grounded FAIL
    ungrounded PASS            a                b
    ungrounded FAIL            c                d

Pairs in `a` and `d` (both right, both wrong) carry no information about which
condition is better, so they drop out. The question becomes: of the b+c pairs
where exactly one condition succeeded, is the split more lopsided than a coin
flip would plausibly produce? That is a binomial test on b out of (b+c).

The EXACT version is used rather than the chi-square approximation because with
a probe set this small (tens of pairs, often fewer than 25 discordant) the
approximation is unreliable. Exact is also simply correct at any n, so there is
no reason to switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass
class McNemarResult:
    b: int              # condition A passed, condition B failed
    c: int              # condition A failed, condition B passed
    n_discordant: int
    p_value: float
    concordant_pass: int = 0
    concordant_fail: int = 0

    @property
    def n_pairs(self) -> int:
        return self.n_discordant + self.concordant_pass + self.concordant_fail

    def verdict(self, alpha: float = 0.05) -> str:
        if self.n_discordant == 0:
            return "no discordant pairs — the conditions never disagreed"
        direction = "B better than A" if self.c > self.b else "A better than B"
        if self.b == self.c:
            direction = "no difference in direction"
        sig = "significant" if self.p_value < alpha else "NOT significant"
        return f"{direction}, {sig} at alpha={alpha} (p={self.p_value:.4f})"

    def summary_line(self, label_a: str = "A", label_b: str = "B") -> str:
        return (
            f"{label_a} vs {label_b}: {self.n_pairs} pairs, "
            f"{self.n_discordant} discordant "
            f"({label_a}-only wins {self.b}, {label_b}-only wins {self.c}), "
            f"p={self.p_value:.4f}"
        )


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for a fair-coin null.

    For p=0.5 the distribution is symmetric, so the two-sided p-value is simply
    twice the smaller tail, capped at 1.0.
    """
    if n == 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k={k} out of range for n={n}")
    tail = min(k, n - k)
    cumulative = sum(comb(n, i) for i in range(tail + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * cumulative)


def mcnemar_exact(
    outcomes_a: list[bool], outcomes_b: list[bool]
) -> McNemarResult:
    """Exact McNemar test on paired pass/fail outcomes.

    outcomes_a[i] and outcomes_b[i] must be the SAME probe on the SAME image
    under two different conditions. True = the condition behaved acceptably
    (not sycophantic, not hallucinated).
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError(
            f"Paired test needs equal-length outcome lists; "
            f"got {len(outcomes_a)} and {len(outcomes_b)}. "
            "A probe missing from one condition must be dropped from both."
        )

    b = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x and not y)
    c = sum(1 for x, y in zip(outcomes_a, outcomes_b) if not x and y)
    both_pass = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x and y)
    both_fail = sum(1 for x, y in zip(outcomes_a, outcomes_b) if not x and not y)

    n = b + c
    p = exact_binomial_two_sided(b, n) if n else 1.0

    return McNemarResult(
        b=b, c=c, n_discordant=n, p_value=p,
        concordant_pass=both_pass, concordant_fail=both_fail,
    )


def pair_outcomes(
    rows_a: dict[tuple, bool], rows_b: dict[tuple, bool]
) -> tuple[list[bool], list[bool], list[tuple]]:
    """Align two condition dicts keyed by (probe_id, image) into paired lists.

    Keys present in only one condition are dropped — an unpaired observation
    cannot contribute to a paired test, and silently treating it as a failure
    would fabricate signal. The dropped keys are returned so the caller can
    report how many were lost rather than hiding it.
    """
    shared = sorted(set(rows_a) & set(rows_b))
    dropped = sorted((set(rows_a) ^ set(rows_b)))
    return [rows_a[k] for k in shared], [rows_b[k] for k in shared], dropped
