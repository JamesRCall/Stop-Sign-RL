"""Pure, testable attack-objective logic for traffic-sign experiments.

The environment owns rendering and reinforcement-learning state.  This module
owns the definition of *what counts as an attack*, which keeps success criteria
identical across PPO, baselines, angle sweeps, and physical-image evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ATTACK_MODES = (
    "disappearance",
    "untargeted_misclassification",
    "targeted_misclassification",
)


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return IoU for two ``xyxy`` boxes, or zero for malformed boxes."""
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
        bx1, by1, bx2, by2 = [float(v) for v in box_b]
    except (TypeError, ValueError):
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


@dataclass(frozen=True)
class AttackObjectiveConfig:
    """Success criteria shared by all attack modes."""

    mode: str = "disappearance"
    source_conf_threshold: float = 0.20
    target_conf_threshold: float = 0.40
    min_success_rate: float = 0.80
    localization_iou_threshold: float = 0.30
    require_source_suppression: bool = True
    allowed_alternative_class_ids: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if self.mode not in ATTACK_MODES:
            raise ValueError(f"mode must be one of {', '.join(ATTACK_MODES)}")
        for name in ("source_conf_threshold", "target_conf_threshold", "min_success_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < float(self.localization_iou_threshold) <= 1.0:
            raise ValueError("localization_iou_threshold must be in (0, 1]")
        if self.allowed_alternative_class_ids is not None:
            normalized = tuple(int(v) for v in self.allowed_alternative_class_ids)
            if len(set(normalized)) != len(normalized):
                raise ValueError("allowed_alternative_class_ids must not contain duplicates")


@dataclass(frozen=True)
class TransformAttackMetrics:
    """Attack measurements for one expectation-over-transformation sample."""

    source_conf: float = 0.0
    source_iou: float = 0.0
    top_class: Optional[int] = None
    top_conf: float = 0.0
    top_iou: float = 0.0
    alternative_class: Optional[int] = None
    alternative_conf: float = 0.0
    alternative_iou: float = 0.0
    attack_target_conf: float = 0.0
    attack_target_iou: float = 0.0
    disappearance_success: bool = False
    untargeted_success: bool = False
    targeted_success: bool = False


@dataclass(frozen=True)
class AggregateAttackMetrics:
    """Mean metrics and empirical success rates across transformations."""

    n: int = 0
    mean_source_conf: float = 0.0
    mean_source_iou: float = 0.0
    mean_top_conf: float = 0.0
    mean_top_iou: float = 0.0
    mean_alternative_conf: float = 0.0
    mean_alternative_iou: float = 0.0
    mean_attack_target_conf: float = 0.0
    mean_attack_target_iou: float = 0.0
    mean_alternative_margin: float = 0.0
    mean_target_margin: float = 0.0
    disappearance_rate: float = 0.0
    untargeted_rate: float = 0.0
    targeted_rate: float = 0.0
    top_class_counts: Optional[Dict[int, int]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n": int(self.n),
            "mean_source_conf": float(self.mean_source_conf),
            "mean_source_iou": float(self.mean_source_iou),
            "mean_top_conf": float(self.mean_top_conf),
            "mean_top_iou": float(self.mean_top_iou),
            "mean_alternative_conf": float(self.mean_alternative_conf),
            "mean_alternative_iou": float(self.mean_alternative_iou),
            "mean_attack_target_conf": float(self.mean_attack_target_conf),
            "mean_attack_target_iou": float(self.mean_attack_target_iou),
            "mean_alternative_margin": float(self.mean_alternative_margin),
            "mean_target_margin": float(self.mean_target_margin),
            "disappearance_rate": float(self.disappearance_rate),
            "untargeted_rate": float(self.untargeted_rate),
            "targeted_rate": float(self.targeted_rate),
            "top_class_counts": dict(self.top_class_counts or {}),
        }


def summarize_detection(
    detection: Mapping[str, Any],
    sign_box: Sequence[float],
    source_class_id: int,
    attack_target_id: Optional[int],
    config: AttackObjectiveConfig,
) -> TransformAttackMetrics:
    """Summarize detections that spatially correspond to the rendered sign.

    Full-scene top detections are not valid evidence of sign misclassification:
    a person or car elsewhere in the background would otherwise count as success.
    Only boxes whose IoU with the known rendered sign exceeds the configured
    localization threshold participate in classification metrics.
    """
    boxes = list(detection.get("boxes", []) or [])
    confs = list(detection.get("confs", []) or [])
    classes = list(detection.get("clss", []) or [])
    count = min(len(boxes), len(confs), len(classes))

    localized: List[Tuple[float, int, float]] = []
    source_iou = 0.0
    for idx in range(count):
        try:
            conf = float(confs[idx])
            class_id = int(classes[idx])
            overlap = box_iou(boxes[idx], sign_box)
        except (TypeError, ValueError):
            continue
        if class_id == int(source_class_id):
            source_iou = max(source_iou, overlap)
        if overlap >= float(config.localization_iou_threshold):
            localized.append((conf, class_id, overlap))

    localized.sort(key=lambda row: row[0], reverse=True)
    top_conf, top_class, top_iou = (localized[0] if localized else (0.0, None, 0.0))

    source_rows = [row for row in localized if row[1] == int(source_class_id)]
    source_conf = max((row[0] for row in source_rows), default=0.0)

    allowed_alternatives = (
        set(int(v) for v in config.allowed_alternative_class_ids)
        if config.allowed_alternative_class_ids is not None
        else None
    )
    alternative_rows = [
        row
        for row in localized
        if row[1] != int(source_class_id)
        and (allowed_alternatives is None or row[1] in allowed_alternatives)
    ]
    alt_conf, alt_class, alt_iou = (
        max(alternative_rows, key=lambda row: row[0])
        if alternative_rows
        else (0.0, None, 0.0)
    )

    target_rows = (
        [row for row in localized if row[1] == int(attack_target_id)]
        if attack_target_id is not None
        else []
    )
    target_conf, _, target_iou = (
        max(target_rows, key=lambda row: row[0])
        if target_rows
        else (0.0, None, 0.0)
    )

    disappearance = source_conf <= float(config.source_conf_threshold)
    untargeted = bool(
        alt_class is not None
        and alt_conf >= float(config.target_conf_threshold)
        and alt_conf >= source_conf
        and (
            not bool(config.require_source_suppression)
            or source_conf <= float(config.source_conf_threshold)
        )
    )
    targeted = bool(
        attack_target_id is not None
        and top_class is not None
        and int(top_class) == int(attack_target_id)
        and target_conf >= float(config.target_conf_threshold)
        and (
            not bool(config.require_source_suppression)
            or source_conf <= float(config.source_conf_threshold)
        )
    )

    return TransformAttackMetrics(
        source_conf=float(source_conf),
        source_iou=float(source_iou),
        top_class=(int(top_class) if top_class is not None else None),
        top_conf=float(top_conf),
        top_iou=float(top_iou),
        alternative_class=(int(alt_class) if alt_class is not None else None),
        alternative_conf=float(alt_conf),
        alternative_iou=float(alt_iou),
        attack_target_conf=float(target_conf),
        attack_target_iou=float(target_iou),
        disappearance_success=bool(disappearance),
        untargeted_success=bool(untargeted),
        targeted_success=bool(targeted),
    )


def aggregate_metrics(rows: Iterable[TransformAttackMetrics]) -> AggregateAttackMetrics:
    """Aggregate per-transform measurements with no hidden weighting."""
    values = list(rows)
    if not values:
        return AggregateAttackMetrics(top_class_counts={})

    def mean(items: Iterable[float]) -> float:
        seq = [float(v) for v in items]
        return float(sum(seq) / len(seq)) if seq else 0.0

    counts: Dict[int, int] = {}
    for row in values:
        if row.top_class is not None:
            counts[int(row.top_class)] = counts.get(int(row.top_class), 0) + 1

    return AggregateAttackMetrics(
        n=len(values),
        mean_source_conf=mean(row.source_conf for row in values),
        mean_source_iou=mean(row.source_iou for row in values),
        mean_top_conf=mean(row.top_conf for row in values),
        mean_top_iou=mean(row.top_iou for row in values),
        mean_alternative_conf=mean(row.alternative_conf for row in values),
        mean_alternative_iou=mean(row.alternative_iou for row in values),
        mean_attack_target_conf=mean(row.attack_target_conf for row in values),
        mean_attack_target_iou=mean(row.attack_target_iou for row in values),
        mean_alternative_margin=mean(
            row.alternative_conf - row.source_conf for row in values
        ),
        mean_target_margin=mean(
            row.attack_target_conf - row.source_conf for row in values
        ),
        disappearance_rate=mean(row.disappearance_success for row in values),
        untargeted_rate=mean(row.untargeted_success for row in values),
        targeted_rate=mean(row.targeted_success for row in values),
        top_class_counts=counts,
    )


def attack_success(metrics: AggregateAttackMetrics, config: AttackObjectiveConfig) -> bool:
    """Apply the selected mode's explicit EOT success-rate criterion."""
    rates = {
        "disappearance": metrics.disappearance_rate,
        "untargeted_misclassification": metrics.untargeted_rate,
        "targeted_misclassification": metrics.targeted_rate,
    }
    return bool(metrics.n > 0 and rates[config.mode] >= float(config.min_success_rate))


def reward_terms(
    metrics: AggregateAttackMetrics,
    config: AttackObjectiveConfig,
) -> Tuple[float, float]:
    """Return ``(classification_gain, localization_gain)`` for the mode.

    The first term combines a sparse empirical success rate with a dense class
    margin.  The second term has mode-aware direction: box drift helps a
    disappearance attack, whereas overlap with the physical sign is required to
    substantiate a misclassification claim.
    """
    if config.mode == "disappearance":
        return 0.0, max(0.0, 1.0 - float(metrics.mean_source_iou))
    if config.mode == "untargeted_misclassification":
        classification = float(metrics.untargeted_rate) + max(
            0.0, float(metrics.mean_alternative_margin)
        )
        return classification, float(metrics.mean_alternative_iou)
    classification = float(metrics.targeted_rate) + max(
        0.0, float(metrics.mean_target_margin)
    )
    return classification, float(metrics.mean_attack_target_iou)


def success_progress(
    metrics: AggregateAttackMetrics,
    config: AttackObjectiveConfig,
) -> float:
    """Return a signed, smooth progress signal around the success boundary."""
    if config.mode == "disappearance":
        return float(config.source_conf_threshold) - float(metrics.mean_source_conf)
    if config.mode == "untargeted_misclassification":
        return (
            float(metrics.untargeted_rate)
            - float(config.min_success_rate)
            + 0.25 * float(metrics.mean_alternative_margin)
        )
    return (
        float(metrics.targeted_rate)
        - float(config.min_success_rate)
        + 0.25 * float(metrics.mean_target_margin)
    )
