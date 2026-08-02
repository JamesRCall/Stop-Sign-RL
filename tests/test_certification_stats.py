import math
from fractions import Fraction

import pytest

from utils.certification_stats import (
    binomial_lower_tail,
    binomial_upper_tail,
    bonferroni_local_alpha,
    clopper_pearson_lower,
    clopper_pearson_upper,
)


def test_binomial_tails_match_exact_small_distribution():
    # P[Binomial(10, 0.5) >= 7] = (120 + 45 + 10 + 1) / 1024.
    assert binomial_upper_tail(7, 10, 0.5) == pytest.approx(0.171875, abs=1e-14)
    assert binomial_lower_tail(3, 10, 0.5) == pytest.approx(0.171875, abs=1e-14)
    assert binomial_upper_tail(0, 10, 0.2) == 1.0
    assert binomial_upper_tail(11, 10, 0.2) == 0.0


def test_exact_bounds_match_known_clopper_pearson_values():
    assert clopper_pearson_lower(5, 10, 0.05) == pytest.approx(
        0.2224411010081294, abs=2e-14
    )
    assert clopper_pearson_upper(5, 10, 0.05) == pytest.approx(
        0.7775588989918706, abs=2e-14
    )
    assert clopper_pearson_lower(9, 10, 0.05) == pytest.approx(
        0.6058366975634952, abs=2e-14
    )


def test_exact_bound_endpoints_and_all_success_formula():
    assert clopper_pearson_lower(0, 10, 0.05) == 0.0
    assert clopper_pearson_upper(10, 10, 0.05) == 1.0
    assert clopper_pearson_lower(10, 10, 0.05) == pytest.approx(
        math.pow(0.05, 0.1), abs=2e-14
    )
    assert clopper_pearson_upper(0, 10, 0.05) == pytest.approx(
        1.0 - math.pow(0.05, 0.1), abs=2e-14
    )


def test_bounds_are_monotone_in_success_count():
    lowers = [clopper_pearson_lower(x, 25, 0.01) for x in range(26)]
    uppers = [clopper_pearson_upper(x, 25, 0.01) for x in range(26)]
    assert lowers == sorted(lowers)
    assert uppers == sorted(uppers)


def test_tail_stays_stable_for_large_high_success_problem():
    value = binomial_upper_tail(990, 1000, 0.95)
    assert 0.0 < value < 1.0
    lower = clopper_pearson_lower(990, 1000, 1e-6)
    assert 0.95 < lower < 0.99


def test_bonferroni_allocation_is_exact_rational_arithmetic():
    assert bonferroni_local_alpha(Fraction(1, 20), 12) == Fraction(1, 240)
    with pytest.raises(ValueError, match="comparison_count"):
        bonferroni_local_alpha(Fraction(1, 20), 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: clopper_pearson_lower(-1, 10, 0.05),
        lambda: clopper_pearson_upper(11, 10, 0.05),
        lambda: clopper_pearson_lower(1, 0, 0.05),
        lambda: clopper_pearson_lower(1, 10, 0.0),
        lambda: clopper_pearson_lower(True, 10, 0.05),
    ],
)
def test_invalid_bound_inputs_fail_closed(call):
    with pytest.raises((TypeError, ValueError)):
        call()
