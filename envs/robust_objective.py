"""Risk aggregation for one stencil evaluated on several support scenes.

The functions in this module are deliberately independent of Gym and SB3 so
the statistical definition used by training can be unit-tested and reused by
evaluation scripts.  CVaR is computed over the empirical distribution with a
fractional boundary sample, rather than rounding the requested tail size.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


def _finite_values(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("risk aggregation received a non-finite value")
        out.append(number)
    if not out:
        raise ValueError("risk aggregation requires at least one value")
    return out


def empirical_cvar(
    values: Iterable[float],
    *,
    alpha: float = 0.25,
    tail: str = "lower",
) -> float:
    """Return empirical CVaR for the requested probability mass.

    ``lower`` is appropriate for rewards and margins where small values are
    harmful.  ``upper`` is appropriate for losses such as inactive-state
    confidence drop.  When ``alpha * n`` is fractional, the boundary sample is
    included with exactly the remaining mass.
    """
    probability = float(alpha)
    if not 0.0 < probability <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    direction = str(tail).strip().lower()
    if direction not in ("lower", "upper"):
        raise ValueError("tail must be 'lower' or 'upper'")

    ordered = sorted(_finite_values(values), reverse=(direction == "upper"))
    mass = probability * len(ordered)
    full = int(math.floor(mass))
    remainder = mass - full
    weighted_sum = sum(ordered[:full])
    if remainder > 1e-12:
        weighted_sum += remainder * ordered[full]
    return float(weighted_sum / mass)


@dataclass(frozen=True)
class SupportRiskSummary:
    """Reviewer-facing summary of a persistent stencil's support batch."""

    support_count: int
    risk_alpha: float
    reward_lower_cvar: float
    reward_mean: float
    joint_success_rate: float
    source_conf_upper_cvar: float
    target_margin_lower_cvar: float
    day_drop_upper_cvar: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "support_count": int(self.support_count),
            "risk_alpha": float(self.risk_alpha),
            "reward_lower_cvar": float(self.reward_lower_cvar),
            "reward_mean": float(self.reward_mean),
            "joint_success_rate": float(self.joint_success_rate),
            "source_conf_upper_cvar": float(self.source_conf_upper_cvar),
            "target_margin_lower_cvar": float(self.target_margin_lower_cvar),
            "day_drop_upper_cvar": float(self.day_drop_upper_cvar),
        }


def summarize_support_risk(
    rewards: Sequence[float],
    infos: Sequence[Mapping[str, Any]],
    *,
    alpha: float = 0.25,
) -> SupportRiskSummary:
    """Aggregate per-support rewards and explicit attack measurements.

    Missing optional metrics are represented by zero rather than silently
    dropping a support scene.  This makes the support count and denominators
    stable and prevents favorable filtering during training.
    """
    if len(rewards) != len(infos):
        raise ValueError("rewards and infos must have the same length")
    reward_values = _finite_values(rewards)

    def metric(name: str, default: float = 0.0) -> list[float]:
        return [float(info.get(name, default)) for info in infos]

    successes = [1.0 if bool(info.get("attack_success", False)) else 0.0 for info in infos]
    source_conf = metric("mean_source_conf")
    target_margin = [
        float(info.get("mean_target_margin", info.get("mean_alternative_margin", 0.0)))
        for info in infos
    ]
    day_drop = metric("drop_day")
    return SupportRiskSummary(
        support_count=len(reward_values),
        risk_alpha=float(alpha),
        reward_lower_cvar=empirical_cvar(reward_values, alpha=alpha, tail="lower"),
        reward_mean=float(sum(reward_values) / len(reward_values)),
        joint_success_rate=float(sum(successes) / len(successes)),
        source_conf_upper_cvar=empirical_cvar(source_conf, alpha=alpha, tail="upper"),
        target_margin_lower_cvar=empirical_cvar(
            target_margin, alpha=alpha, tail="lower"
        ),
        day_drop_upper_cvar=empirical_cvar(day_drop, alpha=alpha, tail="upper"),
    )
