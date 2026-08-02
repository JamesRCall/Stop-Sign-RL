"""Exact finite-sample binomial bounds for risk certification.

The functions in this module intentionally use only the Python standard
library.  One-sided Clopper--Pearson limits are obtained by inverting stable
binomial tails.  The implementation never subtracts a probability close to
one directly: complementary tails use ``expm1`` in log space instead.

These bounds assume independent Bernoulli sampling within each claimed
population.  Bonferroni allocation supplies simultaneous coverage across an
arbitrary, preregistered family of claims; it does not repair violations of the
within-claim sampling assumption.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Tuple


_NEGATIVE_INFINITY = float("-inf")


def _require_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _validate_binomial_counts(successes: int, trials: int) -> Tuple[int, int]:
    x = _require_int(successes, name="successes")
    n = _require_int(trials, name="trials")
    if n <= 0:
        raise ValueError("trials must be > 0")
    if x < 0 or x > n:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    return x, n


def _validate_probability(value: float, *, name: str, strict: bool = False) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number") from exc
    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite")
    if strict:
        if not 0.0 < probability < 1.0:
            raise ValueError(f"{name} must be in (0, 1)")
    elif not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _logaddexp(left: float, right: float) -> float:
    if left == _NEGATIVE_INFINITY:
        return right
    if right == _NEGATIVE_INFINITY:
        return left
    high, low = (left, right) if left >= right else (right, left)
    return high + math.log1p(math.exp(low - high))


def _log_binomial_probability_range(
    trials: int,
    probability: float,
    start: int,
    stop: int,
) -> float:
    """Return log(sum(P[X=k], k=start..stop)) for a binomial X."""
    n = int(trials)
    p = float(probability)
    if start > stop:
        return _NEGATIVE_INFINITY
    if p == 0.0:
        return 0.0 if start <= 0 <= stop else _NEGATIVE_INFINITY
    if p == 1.0:
        return 0.0 if start <= n <= stop else _NEGATIVE_INFINITY

    log_p = math.log(p)
    log_q = math.log1p(-p)
    k = int(start)
    log_term = (
        math.lgamma(n + 1.0)
        - math.lgamma(k + 1.0)
        - math.lgamma(n - k + 1.0)
        + k * log_p
        + (n - k) * log_q
    )
    log_total = log_term
    log_odds = log_p - log_q
    while k < int(stop):
        # P(X=k+1) / P(X=k) = ((n-k)/(k+1)) * p/(1-p).
        log_term += math.log(n - k) - math.log(k + 1) + log_odds
        k += 1
        log_total = _logaddexp(log_total, log_term)
    return log_total


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    """Return ``P[X >= successes]`` for ``X ~ Binomial(trials, probability)``.

    The shorter side of the distribution is summed in log space.  When the
    complementary lower tail is shorter, ``-expm1(log_cdf)`` avoids catastrophic
    cancellation when the desired upper tail is small.
    """
    k = _require_int(successes, name="successes")
    n = _require_int(trials, name="trials")
    p = _validate_probability(probability, name="probability")
    if n < 0:
        raise ValueError("trials must be >= 0")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    # Sum the probabilistically smaller side directly.  Choosing only by the
    # number of terms is inaccurate for extreme confidence levels: a short CDF
    # can still be 1-alpha, and recovering alpha from it loses relative digits.
    # The threshold/mean relationship reliably identifies the small side of a
    # unimodal binomial distribution for the tail inversions used below.
    if float(k) > float(n) * p:
        log_tail = _log_binomial_probability_range(n, p, k, n)
        value = 0.0 if log_tail == _NEGATIVE_INFINITY else math.exp(log_tail)
    else:
        log_complement = _log_binomial_probability_range(n, p, 0, k - 1)
        value = (
            1.0
            if log_complement == _NEGATIVE_INFINITY
            else -math.expm1(min(0.0, log_complement))
        )
    return min(1.0, max(0.0, float(value)))


def binomial_lower_tail(successes: int, trials: int, probability: float) -> float:
    """Return ``P[X <= successes]`` for a binomial random variable."""
    k = _require_int(successes, name="successes")
    n = _require_int(trials, name="trials")
    p = _validate_probability(probability, name="probability")
    if n < 0:
        raise ValueError("trials must be >= 0")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # If Y=n-X, then Y ~ Binomial(n, 1-p) and X<=k iff Y>=n-k.
    return binomial_upper_tail(n - k, n, 1.0 - p)


def clopper_pearson_lower(
    successes: int,
    trials: int,
    alpha: float,
    *,
    iterations: int = 80,
) -> float:
    """Return a conservative one-sided exact lower confidence bound.

    The result inverts ``P_p[X >= successes] = alpha``.  The returned bisection
    endpoint is deliberately on the conservative (lower) side of the root.
    """
    x, n = _validate_binomial_counts(successes, trials)
    tail_alpha = _validate_probability(alpha, name="alpha", strict=True)
    count = _require_int(iterations, name="iterations")
    if count < 1:
        raise ValueError("iterations must be >= 1")
    if x == 0:
        return 0.0

    low = 0.0
    high = 1.0
    for _ in range(count):
        middle = (low + high) / 2.0
        if middle == low or middle == high:
            break
        if binomial_upper_tail(x, n, middle) >= tail_alpha:
            high = middle
        else:
            low = middle
    return float(low)


def clopper_pearson_upper(
    successes: int,
    trials: int,
    alpha: float,
    *,
    iterations: int = 80,
) -> float:
    """Return a conservative one-sided exact upper confidence bound."""
    x, n = _validate_binomial_counts(successes, trials)
    tail_alpha = _validate_probability(alpha, name="alpha", strict=True)
    count = _require_int(iterations, name="iterations")
    if count < 1:
        raise ValueError("iterations must be >= 1")
    if x == n:
        return 1.0

    low = 0.0
    high = 1.0
    for _ in range(count):
        middle = (low + high) / 2.0
        if middle == low or middle == high:
            break
        if binomial_lower_tail(x, n, middle) >= tail_alpha:
            low = middle
        else:
            high = middle
    return float(high)


def bonferroni_local_alpha(
    familywise_alpha: Fraction,
    comparison_count: int,
) -> Fraction:
    """Return the exact equal Bonferroni allocation ``alpha / m``."""
    if not isinstance(familywise_alpha, Fraction):
        raise TypeError("familywise_alpha must be fractions.Fraction")
    if not 0 < familywise_alpha < 1:
        raise ValueError("familywise_alpha must be in (0, 1)")
    count = _require_int(comparison_count, name="comparison_count")
    if count <= 0:
        raise ValueError("comparison_count must be > 0")
    return familywise_alpha / count


__all__ = [
    "binomial_lower_tail",
    "binomial_upper_tail",
    "bonferroni_local_alpha",
    "clopper_pearson_lower",
    "clopper_pearson_upper",
]
