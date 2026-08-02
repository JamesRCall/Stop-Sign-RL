"""Certify one immutable traffic-sign stencil on fresh RNG seeds.

This evaluator deliberately performs no policy inference, search, or pattern
selection.  It loads one saved pattern, applies the identical set of cells after
every reset, and delegates rendering and joint success semantics to the shared
baseline helpers.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.grid_utils import (  # noqa: E402
    _apply_pattern,
    _default_cfg_for_env,
    _eval_pattern,
    build_env_from_args,
)
from envs.attack_objective import ATTACK_MODES  # noqa: E402


DEFAULT_CERTIFICATION_SEED_BASE = 1_000_000
WILSON_95_Z = 1.959963984540054


@dataclass(frozen=True)
class FrozenPattern:
    """One validated saved stencil and its provenance."""

    pattern_type: str
    values: Tuple[int, ...]
    locator: str
    source_seed: Optional[int]
    declared_grid_cell_px: Optional[int]
    trace_paint_name: Optional[str]
    declared_config: Mapping[str, Any]
    input_sha256: str
    pattern_sha256: str


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Pattern file is not valid JSON: {path}: {exc}") from exc


def _validated_integer_list(value: Any, *, locator: str) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{locator} must be a JSON list of integer cell indices")

    out: List[int] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(
                f"{locator}[{position}] must be an integer, got {item!r}"
            )
        out.append(int(item))

    if not out:
        raise ValueError(
            f"{locator} is empty; a frozen attack stencil must contain at least one cell"
        )
    if len(set(out)) != len(out):
        raise ValueError(
            f"{locator} contains duplicate cells; refusing to alter the saved stencil"
        )
    return tuple(out)


def _optional_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pattern_digest(pattern_type: str, values: Sequence[int]) -> str:
    canonical = json.dumps(
        {"pattern_type": str(pattern_type), "values": list(values)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _json_digest(value: Any) -> str:
    canonical = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _image_digest(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"mode": str(image.mode), "size": [int(v) for v in image.size]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(image.tobytes())
    return digest.hexdigest()


def _applied_stencil_identity(env: Any) -> Dict[str, Any]:
    selected_indices = [
        index
        for index, selected in enumerate(env._episode_cells.reshape(-1).tolist())
        if bool(selected)
    ]
    canonical_mask = {
        "grid_shape": [int(env.Gh), int(env.Gw)],
        "selected_indices": selected_indices,
    }
    paint = getattr(env, "paint", None)
    paint_descriptor = {
        key: getattr(paint, key, None)
        for key in (
            "name",
            "day_hex",
            "active_hex",
            "translucent",
            "day_alpha",
            "active_alpha",
        )
    }
    valid_mask_digest = hashlib.sha256(env._valid_cells.tobytes()).hexdigest()
    context = {
        **canonical_mask,
        "grid_cell_px": int(env.grid_cell_px),
        "cell_cover_thresh": float(env.cell_cover_thresh),
        "valid_cell_mask_sha256": valid_mask_digest,
        "sign_alpha_sha256": _image_digest(env._sign_alpha),
        "sign_day_rgba_sha256": _image_digest(env.sign_rgba_day),
        "sign_active_rgba_sha256": _image_digest(env.sign_rgba_on),
        "paint": paint_descriptor,
        "paint_action_mode": str(getattr(env, "paint_action_mode", "fixed")),
    }
    if str(getattr(env, "paint_action_mode", "fixed")) == "joint_palette":
        assignments = env._selected_material_assignments()
        palette = [env._paint_descriptor(value) for value in env.paint_palette]
        canonical_mask.update(
            {
                "action_encoding": str(env.action_encoding),
                "cell_material_assignments": assignments,
            }
        )
        context.update(
            {
                "action_encoding": str(env.action_encoding),
                "cell_material_assignments": assignments,
                "paint_palette": palette,
                "paint_palette_sha256": _json_digest(palette),
            }
        )
    return {
        **context,
        "canonical_mask_sha256": _json_digest(canonical_mask),
        "physical_stencil_sha256": _json_digest(context),
    }


def _source_seed_from_record(record: Mapping[str, Any]) -> Optional[int]:
    episode_meta = record.get("episode_meta")
    if isinstance(episode_meta, Mapping):
        reset_seed = _optional_int(episode_meta.get("reset_seed"))
        if reset_seed is not None:
            return reset_seed
    for key in ("reset_seed", "scene_seed", "seed"):
        candidate = _optional_int(record.get(key))
        if candidate is not None:
            return candidate
    return None


def _declared_config_from_records(
    records: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
) -> Dict[str, Any]:
    """Collect configuration evidence from summaries and callback records."""
    declared: Dict[str, Any] = {}
    direct_fields = (
        "data",
        "bgdir",
        "bg_mode",
        "no_pole",
        "sign_profile",
        "sign_image",
        "sign_active_image",
        "source_class",
        "attack_mode",
        "attack_target_class",
        "allowed_alternative_classes",
        "success_conf",
        "target_conf",
        "min_attack_success_rate",
        "min_clean_detection_rate",
        "localization_iou",
        "require_source_suppression",
        "require_day_preservation",
        "day_tolerance",
        "area_cap_frac",
        "detector",
        "detector_model",
        "yolo_version",
        "yolo_weights",
        "transform_strength",
        "fixed_angle_deg",
        "grid_cell",
        "cell_cover_thresh",
        "paint",
        "paint_list",
        "paint_action_mode",
        "paint_palette",
        "action_indexing",
    )
    aliases = {
        "objective": "attack_mode",
        "source_class_name": "source_class",
        "attack_target_name": "attack_target_class",
        "success_conf_threshold": "success_conf",
        "target_conf_threshold": "target_conf",
        "localization_iou_threshold": "localization_iou",
        "area_cap": "area_cap_frac",
        "grid_cell_px": "grid_cell",
        "paint_name": "paint",
    }
    for record in records:
        config = record.get("config")
        if isinstance(config, Mapping):
            declared.update(dict(config))
        for field in direct_fields:
            if field in record:
                declared[field] = record[field]
        for source_key, destination_key in aliases.items():
            if source_key in record:
                declared[destination_key] = record[source_key]

    # Per-stencil trace metadata is more precise than a run-level paint list.
    if trace.get("grid_cell_px") is not None:
        declared["grid_cell"] = trace["grid_cell_px"]
    if trace.get("paint_name") not in (None, ""):
        declared["paint"] = trace["paint_name"]
    if "fixed_angle_deg" in trace:
        declared["fixed_angle_deg"] = trace["fixed_angle_deg"]
    return declared


def load_frozen_pattern(
    json_path: os.PathLike[str] | str,
    *,
    requested_type: str = "auto",
    episode_index: int = 0,
) -> FrozenPattern:
    """Load exactly one supported pattern from a JSON artifact.

    Supported locations are ``$.actions``, ``$.selected_indices``,
    ``$.selected_indices_flat``,
    ``$.trace.selected_indices``, ``$.episodes_detail[i].trace.selected_indices``,
    and ``$[i].trace.selected_indices`` for per-episode JSON arrays. Auto mode
    fails when more than one candidate is present so certification never
    silently chooses a different stencil than the reviewer intended.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pattern JSON not found: {path}")
    if requested_type not in ("auto", "actions", "selected_indices"):
        raise ValueError(
            "requested_type must be 'auto', 'actions', or 'selected_indices'"
        )
    if int(episode_index) < 0:
        raise ValueError("episode_index must be >= 0")

    payload = _read_json(path)
    candidates: List[Dict[str, Any]] = []
    nested_error: Optional[str] = None

    if isinstance(payload, list):
        if int(episode_index) >= len(payload):
            nested_error = (
                f"episode_index {episode_index} is outside the root episode list "
                f"(length {len(payload)})"
            )
        else:
            row = payload[int(episode_index)]
            if not isinstance(row, Mapping):
                nested_error = f"$[{episode_index}] must be an object"
            else:
                trace = row.get("trace")
                if not isinstance(trace, Mapping):
                    nested_error = f"$[{episode_index}].trace must be an object"
                elif "selected_indices" not in trace:
                    nested_error = (
                        f"$[{episode_index}].trace.selected_indices is missing"
                    )
                else:
                    candidates.append(
                        {
                            "pattern_type": "selected_indices",
                            "raw": trace["selected_indices"],
                            "locator": f"$[{episode_index}].trace.selected_indices",
                            "source_seed": _source_seed_from_record(row),
                            "trace": dict(trace),
                            "metadata_records": [row],
                        }
                    )
    elif isinstance(payload, Mapping):
        if "actions" in payload:
            candidates.append(
                {
                    "pattern_type": "actions",
                    "raw": payload["actions"],
                    "locator": "$.actions",
                    "source_seed": _source_seed_from_record(payload),
                    "trace": {},
                    "metadata_records": [payload],
                }
            )
        if "selected_indices" in payload:
            candidates.append(
                {
                    "pattern_type": "selected_indices",
                    "raw": payload["selected_indices"],
                    "locator": "$.selected_indices",
                    "source_seed": _source_seed_from_record(payload),
                    "trace": {},
                    "metadata_records": [payload],
                }
            )
        if "selected_indices_flat" in payload:
            candidates.append(
                {
                    "pattern_type": "selected_indices",
                    "raw": payload["selected_indices_flat"],
                    "locator": "$.selected_indices_flat",
                    "source_seed": _source_seed_from_record(payload),
                    "trace": {},
                    "metadata_records": [payload],
                }
            )
        root_trace = payload.get("trace")
        if isinstance(root_trace, Mapping) and "selected_indices" in root_trace:
            candidates.append(
                {
                    "pattern_type": "selected_indices",
                    "raw": root_trace["selected_indices"],
                    "locator": "$.trace.selected_indices",
                    "source_seed": _source_seed_from_record(payload),
                    "trace": dict(root_trace),
                    "metadata_records": [payload],
                }
            )

        episode_rows = payload.get("episodes_detail")
        if episode_rows is not None:
            if not isinstance(episode_rows, list):
                nested_error = "$.episodes_detail must be a JSON list"
            elif int(episode_index) >= len(episode_rows):
                nested_error = (
                    f"episode_index {episode_index} is outside $.episodes_detail "
                    f"(length {len(episode_rows)})"
                )
            else:
                row = episode_rows[int(episode_index)]
                if not isinstance(row, Mapping):
                    nested_error = (
                        f"$.episodes_detail[{episode_index}] must be an object"
                    )
                else:
                    trace = row.get("trace")
                    if not isinstance(trace, Mapping):
                        nested_error = (
                            f"$.episodes_detail[{episode_index}].trace must be an object"
                        )
                    elif "selected_indices" not in trace:
                        nested_error = (
                            f"$.episodes_detail[{episode_index}].trace.selected_indices "
                            "is missing"
                        )
                    else:
                        candidates.append(
                            {
                                "pattern_type": "selected_indices",
                                "raw": trace["selected_indices"],
                                "locator": (
                                    f"$.episodes_detail[{episode_index}].trace."
                                    "selected_indices"
                                ),
                                "source_seed": _source_seed_from_record(row),
                                "trace": dict(trace),
                                "metadata_records": [payload, row],
                            }
                        )
    else:
        raise ValueError(
            "Pattern JSON root must be an object or per-episode list containing "
            "a supported saved stencil"
        )

    if requested_type != "auto":
        candidates = [
            candidate
            for candidate in candidates
            if candidate["pattern_type"] == requested_type
        ]

    if not candidates:
        detail = f" ({nested_error})" if nested_error else ""
        raise ValueError(
            f"No {requested_type if requested_type != 'auto' else 'supported'} "
            f"pattern found in {path}{detail}"
        )
    if len(candidates) > 1:
        locators = ", ".join(str(candidate["locator"]) for candidate in candidates)
        raise ValueError(
            f"Pattern JSON contains multiple candidates: {locators}. "
            "Use --pattern-type or provide a JSON file containing exactly one pattern."
        )

    chosen = candidates[0]
    values = _validated_integer_list(chosen["raw"], locator=chosen["locator"])
    trace = chosen.get("trace", {})
    declared_config = _declared_config_from_records(
        chosen.get("metadata_records", []),
        trace if isinstance(trace, Mapping) else {},
    )
    declared_grid = _optional_int(
        trace.get("grid_cell_px")
        if isinstance(trace, dict) and trace.get("grid_cell_px") is not None
        else declared_config.get("grid_cell")
    )
    input_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return FrozenPattern(
        pattern_type=str(chosen["pattern_type"]),
        values=values,
        locator=str(chosen["locator"]),
        source_seed=chosen.get("source_seed"),
        declared_grid_cell_px=declared_grid,
        trace_paint_name=(
            str(trace.get("paint_name"))
            if isinstance(trace, Mapping) and trace.get("paint_name") not in (None, "")
            else None
        ),
        declared_config=declared_config,
        input_sha256=input_sha256,
        pattern_sha256=_pattern_digest(str(chosen["pattern_type"]), values),
    )


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = WILSON_95_Z,
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    n = int(trials)
    x = int(successes)
    if n <= 0:
        raise ValueError("trials must be > 0")
    if x < 0 or x > n:
        raise ValueError("successes must satisfy 0 <= successes <= trials")
    z2 = float(z) ** 2
    proportion = float(x) / float(n)
    denominator = 1.0 + z2 / float(n)
    center = (proportion + z2 / (2.0 * float(n))) / denominator
    half_width = (
        float(z)
        * math.sqrt(
            proportion * (1.0 - proportion) / float(n)
            + z2 / (4.0 * float(n) ** 2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    metric_names: Iterable[str],
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """Return finite-only population means and standard deviations."""
    means: Dict[str, Optional[float]] = {}
    standard_deviations: Dict[str, Optional[float]] = {}
    for metric in metric_names:
        values = [
            number
            for number in (_finite_float(row.get(metric)) for row in rows)
            if number is not None
        ]
        if not values:
            means[metric] = None
            standard_deviations[metric] = None
            continue
        mean_value = float(sum(values) / len(values))
        variance = float(sum((value - mean_value) ** 2 for value in values) / len(values))
        means[metric] = mean_value
        standard_deviations[metric] = math.sqrt(variance)
    return means, standard_deviations


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return number if math.isfinite(number) else None


def _normalized_profile(value: Any) -> str:
    return str(value or "stop").strip().lower().replace("-", "_")


def _normalized_token(value: Any) -> str:
    text_value = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return " ".join(text_value.split())


def _normalized_paint(value: Any) -> str:
    token = "".join(character for character in _normalized_token(value) if character.isalnum())
    return token[:-4] if token.endswith("glow") else token


def _normalized_path(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return os.path.normcase(os.path.abspath(os.path.expanduser(raw)))


def _asset_reference(config: Mapping[str, Any], *, active: bool) -> str:
    profile = _normalized_profile(config.get("sign_profile"))
    defaults = {
        "stop": ("stop_sign.png", "stop_sign_uv.png"),
        "speed_limit": ("speed_limit_sign.png", "speed_limit_sign_uv.png"),
    }
    day_explicit = str(config.get("sign_image") or "").strip()
    active_explicit = str(config.get("sign_active_image") or "").strip()
    if active:
        if active_explicit:
            selected = active_explicit
        elif day_explicit:
            selected = day_explicit
        elif profile in defaults:
            selected = str(Path(str(config.get("data") or "./data")) / defaults[profile][1])
        else:
            selected = ""
    elif day_explicit:
        selected = day_explicit
    elif profile in defaults:
        selected = str(Path(str(config.get("data") or "./data")) / defaults[profile][0])
    else:
        selected = ""
    return _normalized_path(selected)


def _effective_source_class(config: Mapping[str, Any]) -> str:
    explicit = config.get("source_class")
    if explicit is not None and str(explicit).strip():
        return _normalized_token(explicit)
    defaults = {"stop": "stop sign", "speed_limit": "speed limit sign"}
    return _normalized_token(defaults.get(_normalized_profile(config.get("sign_profile")), ""))


def _effective_paint(
    config: Mapping[str, Any],
    *,
    trace_paint_name: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if trace_paint_name:
        return _normalized_paint(trace_paint_name), None
    listed = [
        _normalized_paint(part)
        for part in str(config.get("paint_list") or "").split(",")
        if str(part).strip()
    ]
    if len(listed) > 1:
        return None, "saved paint_list contains multiple paints but the trace has no paint_name"
    if len(listed) == 1:
        return listed[0], None
    return _normalized_paint(config.get("paint") or "yellow"), None


def _normalized_class_list(value: Any) -> Tuple[str, ...]:
    if value in (None, ""):
        return ()
    raw_values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    return tuple(sorted(_normalized_token(item) for item in raw_values if str(item).strip()))


def _normalized_palette(value: Any) -> Tuple[str, ...]:
    if value in (None, ""):
        return ()
    raw_values = value if isinstance(value, (list, tuple)) else str(value).split(",")
    names = []
    for item in raw_values:
        raw_name = item.get("name") if isinstance(item, Mapping) else item
        if raw_name is not None and str(raw_name).strip():
            names.append(_normalized_paint(raw_name))
    return tuple(names)


def _float_values_match(saved: Any, requested: Any) -> bool:
    if saved is None or requested is None:
        return saved is None and requested is None
    saved_number = _finite_float(saved)
    requested_number = _finite_float(requested)
    return (
        saved_number is not None
        and requested_number is not None
        and math.isclose(saved_number, requested_number, rel_tol=0.0, abs_tol=1e-12)
    )


def pattern_config_mismatch_groups(
    pattern: FrozenPattern,
    effective_config: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Separate immutable-stencil incompatibilities from explicit study transfers."""
    geometry: List[str] = []
    protocol: List[str] = []
    source_config = pattern.declared_config

    def add_mismatch(target: List[str], field: str, saved: Any, requested: Any) -> None:
        target.append(f"{field} saved={saved!r} requested={requested!r}")

    if (
        pattern.declared_grid_cell_px is not None
        and int(pattern.declared_grid_cell_px) != int(effective_config["grid_cell"])
    ):
        add_mismatch(
            geometry,
            "grid_cell",
            pattern.declared_grid_cell_px,
            effective_config["grid_cell"],
        )

    if "sign_profile" in source_config:
        saved_profile = _normalized_profile(source_config.get("sign_profile"))
        requested_profile = _normalized_profile(effective_config.get("sign_profile"))
        if saved_profile != requested_profile:
            add_mismatch(geometry, "sign_profile", saved_profile, requested_profile)

    asset_evidence_keys = {"data", "sign_profile", "sign_image", "sign_active_image"}
    if asset_evidence_keys.intersection(source_config):
        for field, active in (("sign_image", False), ("sign_active_image", True)):
            saved_asset = _asset_reference(source_config, active=active)
            requested_asset = _asset_reference(effective_config, active=active)
            if saved_asset != requested_asset:
                add_mismatch(geometry, field, saved_asset, requested_asset)

    if "cell_cover_thresh" in source_config and not _float_values_match(
        source_config.get("cell_cover_thresh"),
        effective_config.get("cell_cover_thresh"),
    ):
        add_mismatch(
            geometry,
            "cell_cover_thresh",
            source_config.get("cell_cover_thresh"),
            effective_config.get("cell_cover_thresh"),
        )

    saved_paint, paint_error = _effective_paint(
        source_config,
        trace_paint_name=pattern.trace_paint_name,
    )
    requested_paint, requested_paint_error = _effective_paint(effective_config)
    if paint_error:
        geometry.append(f"paint: {paint_error}")
    elif requested_paint_error:
        geometry.append(f"paint: {requested_paint_error}")
    elif ("paint" in source_config or "paint_list" in source_config) and (
        saved_paint != requested_paint
    ):
        add_mismatch(geometry, "paint", saved_paint, requested_paint)

    if "paint_action_mode" in source_config:
        saved_mode = _normalized_token(source_config.get("paint_action_mode"))
        requested_mode = _normalized_token(effective_config.get("paint_action_mode"))
        if saved_mode != requested_mode:
            add_mismatch(geometry, "paint_action_mode", saved_mode, requested_mode)
    if "paint_palette" in source_config:
        saved_palette = _normalized_palette(source_config.get("paint_palette"))
        requested_palette = _normalized_palette(effective_config.get("paint_palette"))
        if saved_palette != requested_palette:
            add_mismatch(geometry, "paint_palette", saved_palette, requested_palette)
    if "action_indexing" in source_config:
        saved_indexing = _normalized_token(source_config.get("action_indexing"))
        requested_indexing = _normalized_token(effective_config.get("action_indexing"))
        if saved_indexing != requested_indexing:
            add_mismatch(geometry, "action_indexing", saved_indexing, requested_indexing)

    if "source_class" in source_config or "sign_profile" in source_config:
        saved_source = _effective_source_class(source_config)
        requested_source = _effective_source_class(effective_config)
        if saved_source != requested_source:
            add_mismatch(protocol, "source_class", saved_source, requested_source)

    token_fields = (
        "attack_mode",
        "attack_target_class",
        "detector",
        "detector_model",
        "bg_mode",
        "area_cap_mode",
    )
    for field in token_fields:
        if field in source_config:
            saved_value = _normalized_token(source_config.get(field))
            requested_value = _normalized_token(effective_config.get(field))
            if saved_value != requested_value:
                add_mismatch(protocol, field, saved_value, requested_value)

    if "allowed_alternative_classes" in source_config:
        saved_alternatives = _normalized_class_list(
            source_config.get("allowed_alternative_classes")
        )
        requested_alternatives = _normalized_class_list(
            effective_config.get("allowed_alternative_classes")
        )
        if saved_alternatives != requested_alternatives:
            add_mismatch(
                protocol,
                "allowed_alternative_classes",
                saved_alternatives,
                requested_alternatives,
            )

    float_fields = (
        "success_conf",
        "target_conf",
        "min_attack_success_rate",
        "min_clean_detection_rate",
        "localization_iou",
        "day_tolerance",
        "area_cap_frac",
        "transform_strength",
        "fixed_angle_deg",
    )
    for field in float_fields:
        if field in source_config and not _float_values_match(
            source_config.get(field),
            effective_config.get(field),
        ):
            add_mismatch(
                protocol,
                field,
                source_config.get(field),
                effective_config.get(field),
            )

    bool_fields = (
        "require_source_suppression",
        "require_day_preservation",
        "no_pole",
    )
    for field in bool_fields:
        if field in source_config:
            saved_value = bool(int(source_config.get(field)))
            requested_value = bool(int(effective_config.get(field)))
            if saved_value != requested_value:
                add_mismatch(protocol, field, saved_value, requested_value)

    if "bgdir" in source_config:
        saved_bgdir = _normalized_path(source_config.get("bgdir"))
        requested_bgdir = _normalized_path(effective_config.get("bgdir"))
        if saved_bgdir != requested_bgdir:
            add_mismatch(protocol, "bgdir", saved_bgdir, requested_bgdir)

    saved_detector = _normalized_token(source_config.get("detector") or "yolo")
    if saved_detector in {"yolo", "ultralytics"} and (
        "yolo_version" in source_config or "yolo_weights" in source_config
    ):
        saved_weights = source_config.get("yolo_weights") or (
            "./weights/yolov8n.pt"
            if str(source_config.get("yolo_version") or "8") == "8"
            else "./weights/yolo11n.pt"
        )
        requested_weights = effective_config.get("yolo_weights") or (
            "./weights/yolov8n.pt"
            if str(effective_config.get("yolo_version") or "8") == "8"
            else "./weights/yolo11n.pt"
        )
        saved_weights_path = _normalized_path(saved_weights)
        requested_weights_path = _normalized_path(requested_weights)
        if saved_weights_path != requested_weights_path:
            add_mismatch(
                protocol,
                "detector_weights",
                saved_weights_path,
                requested_weights_path,
            )
    return {"geometry_material": geometry, "protocol": protocol}


def pattern_config_mismatches(
    pattern: FrozenPattern,
    effective_config: Mapping[str, Any],
) -> List[str]:
    """Return all compatibility differences for callers needing a flat list."""
    groups = pattern_config_mismatch_groups(pattern, effective_config)
    return groups["geometry_material"] + groups["protocol"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Certify exactly one saved traffic-sign stencil on fresh seeds without "
            "policy inference, adaptation, or search."
        )
    )
    parser.add_argument("--pattern-json", required=True, help="Saved pattern/summary JSON.")
    parser.add_argument(
        "--pattern-type",
        choices=["auto", "actions", "selected_indices"],
        default="auto",
        help="Pattern representation; auto fails if the JSON is ambiguous.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode used for episodes_detail[i] or a root per-episode JSON list.",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_CERTIFICATION_SEED_BASE,
        help="First fresh certification seed (default: 1,000,000).",
    )
    parser.add_argument(
        "--allow-seed-overlap",
        action="store_true",
        help="Allow the selected pattern's source seed in the certification set.",
    )
    parser.add_argument(
        "--allow-protocol-transfer",
        action="store_true",
        help=(
            "Allow detector/objective/distribution metadata to differ and label the "
            "result as an intentional transfer study. Sign/grid/paint changes remain errors."
        ),
    )
    parser.add_argument(
        "--out-json",
        default="./_runs/paper_data/frozen_pattern_eval.json",
        help="Certification JSON containing every fresh-RNG-seed row.",
    )

    parser.add_argument("--data", default="./data")
    parser.add_argument("--bgdir", default="./data/backgrounds")
    parser.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    parser.add_argument("--no-pole", action="store_true")
    parser.add_argument(
        "--sign-profile", choices=["stop", "speed_limit", "custom"], default="stop"
    )
    parser.add_argument("--sign-image", default="")
    parser.add_argument("--sign-active-image", default="")
    parser.add_argument("--source-class", default="")

    parser.add_argument("--attack-mode", choices=ATTACK_MODES, default="disappearance")
    parser.add_argument("--attack-target-class", default="")
    parser.add_argument(
        "--allowed-alternative-classes",
        default="",
        help="Comma-separated semantic alternatives for untargeted misclassification.",
    )
    parser.add_argument("--success-conf", type=float, default=0.20)
    parser.add_argument("--target-conf", type=float, default=0.40)
    parser.add_argument("--min-attack-success-rate", type=float, default=0.80)
    parser.add_argument("--min-clean-detection-rate", type=float, default=0.80)
    parser.add_argument("--localization-iou", type=float, default=0.30)
    parser.add_argument("--require-source-suppression", type=int, choices=[0, 1], default=1)
    parser.add_argument("--require-day-preservation", type=int, choices=[0, 1], default=1)
    parser.add_argument("--day-tolerance", type=float, default=0.05)
    parser.add_argument("--area-cap-frac", type=float, default=0.30)
    parser.add_argument("--area-cap-mode", choices=["soft", "hard"], default="soft")

    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--yolo-version", choices=["8", "11"], default="8")
    parser.add_argument(
        "--detector", choices=["yolo", "torchvision", "rtdetr"], default="yolo"
    )
    parser.add_argument("--detector-model", default="")
    parser.add_argument("--detector-device", default=os.getenv("YOLO_DEVICE", "auto"))
    parser.add_argument("--detector-debug", type=int, choices=[0, 1], default=0)

    parser.add_argument("--eval-K", type=int, default=10)
    parser.add_argument("--grid-cell", type=int, default=16)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument("--transform-strength", type=float, default=1.0)
    parser.add_argument("--fixed-angle-deg", type=float, default=None)
    parser.add_argument("--paint", default="yellow")
    parser.add_argument("--paint-list", default="")
    parser.add_argument(
        "--paint-action-mode",
        choices=["fixed", "joint_palette"],
        default="fixed",
    )
    parser.add_argument("--paint-palette", default="")
    parser.add_argument(
        "--action-indexing",
        choices=["valid_cells", "canonical_full_grid"],
        default="valid_cells",
    )
    parser.add_argument("--cell-cover-thresh", type=float, default=0.60)
    parser.add_argument("--obs-size", type=int, default=224)
    parser.add_argument("--obs-margin", type=float, default=0.10)
    parser.add_argument("--obs-include-mask", type=int, choices=[0, 1], default=1)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be > 0")
    if int(args.eval_K) <= 0:
        raise ValueError("--eval-K must be > 0")
    if int(args.grid_cell) <= 0:
        raise ValueError("--grid-cell must be > 0")
    paints = [part.strip() for part in str(args.paint_list).split(",") if part.strip()]
    if len(paints) > 1:
        raise ValueError(
            "Frozen-pattern certification requires one physical paint; "
            "do not pass a multi-value --paint-list"
        )


def run_certification(args: argparse.Namespace) -> Dict[str, Any]:
    """Run deterministic, non-adaptive certification and return its report."""
    _validate_args(args)
    overall_start = time.perf_counter()
    pattern_path = Path(args.pattern_json)
    pattern = load_frozen_pattern(
        pattern_path,
        requested_type=str(args.pattern_type),
        episode_index=int(args.episode_index),
    )

    effective_config = _default_cfg_for_env(vars(args))
    effective_config["seed"] = int(args.seed_base)
    mismatch_groups = pattern_config_mismatch_groups(pattern, effective_config)
    geometry_mismatches = mismatch_groups["geometry_material"]
    protocol_mismatches = mismatch_groups["protocol"]
    mismatches = geometry_mismatches + protocol_mismatches
    if geometry_mismatches:
        raise ValueError(
            "Saved pattern cannot represent the identical physical stencil under the "
            "requested sign/grid/material configuration: "
            + "; ".join(geometry_mismatches)
        )
    if protocol_mismatches and not bool(args.allow_protocol_transfer):
        raise ValueError(
            "Saved detector/objective/distribution metadata differs from certification: "
            + "; ".join(protocol_mismatches)
            + ". Repeat the saved settings, or pass --allow-protocol-transfer for an "
            "explicitly labeled transfer study."
        )

    seeds = [int(args.seed_base) + index for index in range(int(args.episodes))]
    if (
        pattern.source_seed is not None
        and int(pattern.source_seed) in seeds
        and not bool(args.allow_seed_overlap)
    ):
        raise ValueError(
            f"Certification seeds include source seed {pattern.source_seed}; "
            "choose a distinct --seed-base or pass --allow-seed-overlap explicitly"
        )

    setup_start = time.perf_counter()
    env = build_env_from_args(SimpleNamespace(**effective_config))
    setup_runtime = float(time.perf_counter() - setup_start)
    metric_names = (
        "c0_day",
        "c0_on",
        "c_day",
        "c_on",
        "drop_day",
        "drop_on",
        "area_frac",
        "objective_success",
        "clean_eligible",
        "clean_detection_rate",
        "day_correct_rate",
        "eligible_transform_count",
        "total_transform_count",
        "disappearance_success_rate",
        "misclassification_success_rate",
        "targeted_success_rate",
        "mean_source_conf",
        "mean_source_iou",
        "mean_alternative_conf",
        "mean_alternative_iou",
        "mean_attack_target_conf",
        "mean_attack_target_iou",
        "mean_alternative_margin",
        "mean_target_margin",
        "day_preserved",
        "within_area_budget",
    )

    rows: List[Dict[str, Any]] = []
    applied_stencil: Optional[Dict[str, Any]] = None
    certification_start = time.perf_counter()
    try:
        for episode_index, seed in enumerate(seeds):
            episode_start = time.perf_counter()
            env.reset(seed=int(seed))
            clean_queries = int(getattr(env, "_detector_queries", 0))
            _apply_pattern(env, pattern.pattern_type, list(pattern.values))
            selected_cells = int(env._episode_cells.sum())
            if selected_cells != len(pattern.values):
                raise ValueError(
                    "Saved pattern is incompatible with the requested sign/grid: "
                    f"loaded {len(pattern.values)} unique cells but only "
                    f"{selected_cells} valid cells were applied"
                )
            episode_stencil = _applied_stencil_identity(env)
            if applied_stencil is None:
                applied_stencil = episode_stencil
            elif (
                episode_stencil["physical_stencil_sha256"]
                != applied_stencil["physical_stencil_sha256"]
            ):
                raise RuntimeError(
                    "Environment reset changed the resolved physical stencil; "
                    "certification requires one immutable mask and paint"
                )

            raw_metrics = _eval_pattern(env, eval_k=int(args.eval_K))
            total_queries = int(getattr(env, "_detector_queries", 0))
            overlay_queries = total_queries - clean_queries
            row: Dict[str, Any] = {
                "episode_index": int(episode_index),
                "seed": int(seed),
                "success": bool(float(raw_metrics.get("success", 0.0)) >= 0.5),
                "selected_cells": selected_cells,
                "canonical_mask_sha256": episode_stencil["canonical_mask_sha256"],
                "physical_stencil_sha256": episode_stencil[
                    "physical_stencil_sha256"
                ],
                "detector_image_queries": total_queries,
                "clean_baseline_image_queries": clean_queries,
                "overlay_image_queries": overlay_queries,
            }
            for metric in metric_names:
                row[metric] = _finite_float(raw_metrics.get(metric))
            row["runtime_sec"] = float(time.perf_counter() - episode_start)
            rows.append(row)
    finally:
        env.close()

    certification_runtime = float(time.perf_counter() - certification_start)
    successes = sum(1 for row in rows if bool(row["success"]))
    success_rate = float(successes / len(rows))
    ci_lower, ci_upper = wilson_interval(successes, len(rows))
    metric_means, metric_stds = summarize_metric_rows(rows, metric_names)
    total_queries = sum(int(row["detector_image_queries"]) for row in rows)
    mean_runtime = float(sum(float(row["runtime_sec"]) for row in rows) / len(rows))

    report = {
        "schema_version": 1,
        "method": "frozen_pattern_certification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "protocol": {
            "no_policy_inference": True,
            "no_adaptation": True,
            "no_search": True,
            "one_pattern_reused_for_every_seed": True,
            "geometry_material_identity_enforced": True,
            "intentional_protocol_transfer": bool(protocol_mismatches),
            "pattern_application": "baselines.grid_utils._apply_pattern",
            "joint_evaluator": "baselines.grid_utils._eval_pattern",
            "seed_role": "fresh_rng_seed_certification",
            "disjoint_background_dataset_split_enforced": False,
            "query_accounting_scope": (
                "certification_only_excludes_pattern_search_and_training"
            ),
        },
        "pattern": {
            "json_path": str(pattern_path),
            "json_path_abs": str(pattern_path.resolve()),
            "locator": pattern.locator,
            "pattern_type": pattern.pattern_type,
            "values": list(pattern.values),
            "cell_count": len(pattern.values),
            "source_seed": pattern.source_seed,
            "declared_grid_cell_px": pattern.declared_grid_cell_px,
            "trace_paint_name": pattern.trace_paint_name,
            "declared_config": dict(pattern.declared_config),
            "input_sha256": pattern.input_sha256,
            "source_encoding_sha256": pattern.pattern_sha256,
            "pattern_sha256": applied_stencil["physical_stencil_sha256"],
            "applied_stencil": applied_stencil,
            "metadata_mismatches": mismatches,
            "geometry_material_mismatches": geometry_mismatches,
            "protocol_mismatches": protocol_mismatches,
        },
        "episodes": len(rows),
        "seed_base": int(args.seed_base),
        "seed_end_inclusive": int(seeds[-1]),
        "n_success": int(successes),
        "success_rate": success_rate,
        "wilson_95_ci": {"lower": ci_lower, "upper": ci_upper},
        "mean_metrics": metric_means,
        "population_std_metrics": metric_stds,
        "certification_detector_image_queries_total": int(total_queries),
        "certification_detector_image_queries_mean_per_seed": float(
            total_queries / len(rows)
        ),
        # Backward-friendly aliases; protocol.query_accounting_scope defines scope.
        "detector_image_queries_total": int(total_queries),
        "detector_image_queries_mean_per_seed": float(total_queries / len(rows)),
        "runtime": {
            "environment_setup_sec": setup_runtime,
            "certification_sec": certification_runtime,
            "mean_per_seed_sec": mean_runtime,
            "total_sec": float(time.perf_counter() - overall_start),
        },
        "rows": rows,
        "effective_config": effective_config,
        "argv": list(sys.argv),
    }
    return _json_safe(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_certification(args)
    interval = report["wilson_95_ci"]
    pattern = report["pattern"]
    print(
        "[FROZEN] "
        f"type={pattern['pattern_type']} cells={pattern['cell_count']} "
        f"sha256={pattern['pattern_sha256'][:12]}"
    )
    print(
        "[FROZEN] "
        f"success={report['n_success']}/{report['episodes']} "
        f"rate={report['success_rate']:.4f} "
        f"wilson95=[{interval['lower']:.4f}, {interval['upper']:.4f}]"
    )
    print(
        "[FROZEN] "
        f"detector_image_queries={report['detector_image_queries_total']} "
        f"runtime_sec={report['runtime']['certification_sec']:.3f}"
    )

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"[FROZEN] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
