#!/usr/bin/env python3
"""Run every declared detector through one reproducible patch-policy matrix.

The detector checkpoints remain frozen.  "Training a model" in this matrix
means training a separate black-box patch policy against that detector, then
freezing and evaluating the resulting stencil.  Each child run delegates to
``run_simulation_test.sh`` so objectives, baselines, budgets, and artifact
formats stay identical.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DETECTORS = {"yolo", "torchvision", "rtdetr"}
SUPPORTED_MODES = {"targeted_misclassification", "untargeted_misclassification"}
SUPPORTED_PAINTS = {"white", "red", "green", "yellow", "blue", "orange"}
SUPPORTED_TORCHVISION_MODELS = {
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "fcos_resnet50_fpn",
}
SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
MODEL_KEYS = {
    "id",
    "detector",
    "detector_model",
    "yolo_version",
    "weights",
    "enabled",
    "requires_network",
}
# These files can be created by the documented ``nohup`` launch command before
# this process gets a chance to initialize the output directory.  They are
# operational metadata, not experiment artifacts, so they must not make a new
# output directory look like an incomplete run.
EXTERNAL_LAUNCHER_ARTIFACTS = frozenset({"launcher.log", "launcher.pid"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read matrix configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("matrix configuration root must be an object")
    return value


def _strict_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing or unknown:
        raise ValueError(f"{label} invalid fields: missing={missing}, unknown={unknown}")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _finite_fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be finite and in (0, 1]")
    return result


def _resolve_asset(config_path: Path, value: Any, label: str) -> Path:
    raw = Path(_nonempty(value, label))
    result = raw if raw.is_absolute() else config_path.parent / raw
    result = result.resolve()
    if not result.exists():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _resolve_file(config_path: Path, value: Any, label: str) -> Path:
    result = _resolve_asset(config_path, value, label)
    if not result.is_file():
        raise FileNotFoundError(f"{label} must be a file: {result}")
    return result


def _resolve_directory(config_path: Path, value: Any, label: str) -> Path:
    result = _resolve_asset(config_path, value, label)
    if not result.is_dir():
        raise FileNotFoundError(f"{label} must be a directory: {result}")
    return result


def validate_matrix(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    _strict_keys(
        payload,
        {"schema_version", "experiment_id", "description", "attack", "assets", "patch", "models"},
        "$",
    )
    if payload["schema_version"] != 1:
        raise ValueError("schema_version must equal 1")
    _nonempty(payload["experiment_id"], "$.experiment_id")
    _nonempty(payload["description"], "$.description")

    attack = payload["attack"]
    if not isinstance(attack, dict):
        raise ValueError("$.attack must be an object")
    _strict_keys(
        attack,
        {"mode", "source_class", "target_class", "allowed_alternative_classes"},
        "$.attack",
    )
    mode = _nonempty(attack["mode"], "$.attack.mode")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"$.attack.mode must be one of {sorted(SUPPORTED_MODES)}")
    source = _nonempty(attack["source_class"], "$.attack.source_class")
    target = str(attack["target_class"] or "").strip()
    alternatives = str(attack["allowed_alternative_classes"] or "").strip()
    if mode == "targeted_misclassification" and not target:
        raise ValueError("targeted mode requires $.attack.target_class")
    if mode == "targeted_misclassification" and alternatives:
        raise ValueError(
            "targeted mode requires an empty $.attack.allowed_alternative_classes"
        )
    if mode == "untargeted_misclassification" and not alternatives:
        raise ValueError("untargeted mode requires $.attack.allowed_alternative_classes")
    if mode == "untargeted_misclassification" and target:
        raise ValueError("untargeted mode requires an empty $.attack.target_class")
    if target and target.casefold() == source.casefold():
        raise ValueError("source and target classes must differ")

    assets = payload["assets"]
    if not isinstance(assets, dict):
        raise ValueError("$.assets must be an object")
    _strict_keys(assets, {"sign_day", "sign_active", "backgrounds"}, "$.assets")
    sign_day = _resolve_file(path, assets["sign_day"], "$.assets.sign_day")
    sign_active = _resolve_file(path, assets["sign_active"], "$.assets.sign_active")
    backgrounds = _resolve_directory(
        path, assets["backgrounds"], "$.assets.backgrounds"
    )
    if not any(
        entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        for entry in backgrounds.iterdir()
    ):
        raise FileNotFoundError(
            "$.assets.backgrounds contains no supported image files: "
            f"{backgrounds}"
        )
    resolved_assets = {
        "sign_day": str(sign_day),
        "sign_active": str(sign_active),
        "backgrounds": str(backgrounds),
    }

    patch = payload["patch"]
    if not isinstance(patch, dict):
        raise ValueError("$.patch must be an object")
    _strict_keys(
        patch,
        {
            "paint",
            "paint_action_mode",
            "paint_palette",
            "physics",
            "grid_cell",
            "area_cap_fraction",
            "transform_strength",
        },
        "$.patch",
    )
    paint = _nonempty(patch["paint"], "$.patch.paint").lower()
    if paint not in SUPPORTED_PAINTS:
        raise ValueError(f"$.patch.paint must be one of {sorted(SUPPORTED_PAINTS)}")
    paint_action_mode = _nonempty(
        patch["paint_action_mode"], "$.patch.paint_action_mode"
    )
    if paint_action_mode not in {"fixed", "joint_palette"}:
        raise ValueError("$.patch.paint_action_mode must be fixed or joint_palette")
    palette = [
        part.strip().lower()
        for part in str(patch["paint_palette"] or "").split(",")
        if part.strip()
    ]
    if paint_action_mode == "joint_palette":
        if len(palette) < 2:
            raise ValueError("joint_palette requires at least two paint_palette entries")
        if len(palette) != len(set(palette)):
            raise ValueError("paint_palette entries must be unique")
        unknown_paints = sorted(set(palette) - SUPPORTED_PAINTS)
        if unknown_paints:
            raise ValueError("unknown paint_palette entries: " + ", ".join(unknown_paints))
    elif palette:
        raise ValueError("fixed paint_action_mode requires an empty paint_palette")
    physics = _nonempty(patch["physics"], "$.patch.physics")
    if physics not in {"fixed_palette", "synthetic_transport"}:
        raise ValueError("$.patch.physics must be fixed_palette or synthetic_transport")
    if paint_action_mode == "joint_palette" and physics != "fixed_palette":
        raise ValueError("joint_palette currently requires fixed_palette physics")
    grid_cell = patch["grid_cell"]
    if isinstance(grid_cell, bool) or not isinstance(grid_cell, int) or grid_cell < 1:
        raise ValueError("$.patch.grid_cell must be a positive integer")
    _finite_fraction(patch["area_cap_fraction"], "$.patch.area_cap_fraction")
    _finite_fraction(patch["transform_strength"], "$.patch.transform_strength")

    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("$.models must be a non-empty array")
    ids: set[str] = set()
    normalized_models = []
    for index, raw in enumerate(models):
        label = f"$.models[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        _strict_keys(raw, MODEL_KEYS, label)
        model_id = _nonempty(raw["id"], f"{label}.id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", model_id):
            raise ValueError(f"{label}.id is not a safe result identifier")
        if model_id in ids:
            raise ValueError(f"duplicate model id {model_id!r}")
        ids.add(model_id)
        detector = _nonempty(raw["detector"], f"{label}.detector").lower()
        if detector not in SUPPORTED_DETECTORS:
            raise ValueError(f"{label}.detector must be one of {sorted(SUPPORTED_DETECTORS)}")
        if not isinstance(raw["enabled"], bool) or not isinstance(raw["requires_network"], bool):
            raise ValueError(f"{label}.enabled/requires_network must be booleans")
        yolo_version = str(raw["yolo_version"])
        if yolo_version not in {"8", "11"}:
            raise ValueError(f"{label}.yolo_version must be '8' or '11'")
        detector_model = str(raw["detector_model"] or "").strip()
        if detector == "yolo":
            if detector_model:
                raise ValueError(
                    f"{label}.detector_model must be empty for yolo (weights select the model)"
                )
            weights = str(_resolve_file(path, raw["weights"], f"{label}.weights"))
        else:
            if raw["weights"] not in (None, ""):
                raise ValueError(
                    f"{label}.weights must be null for {detector}; this runner would ignore it"
                )
            weights = ""
            if not detector_model:
                raise ValueError(f"{label}.detector_model is required for {detector}")
            if (
                detector == "torchvision"
                and detector_model not in SUPPORTED_TORCHVISION_MODELS
            ):
                raise ValueError(
                    f"{label}.detector_model must be one of "
                    f"{sorted(SUPPORTED_TORCHVISION_MODELS)}"
                )
        normalized_models.append(
            {
                **raw,
                "detector": detector,
                "detector_model": detector_model,
                "yolo_version": yolo_version,
                "weights": weights,
            }
        )

    return {**payload, "assets": resolved_assets, "models": normalized_models}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate one black-box pixel-patch policy for every declared detector model."
    )
    parser.add_argument("--config", default="configs/misclassification_models.json")
    parser.add_argument("--output", default="runs/misclassification_matrix")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0 1 2 3 4")
    parser.add_argument("--model-ids", default="", help="Optional comma-separated subset.")
    parser.add_argument("--max-steps", type=int, default=800_000)
    parser.add_argument("--minimum-steps", type=int, default=50_000)
    parser.add_argument("--early-stop-success-rate", type=float, default=0.80)
    parser.add_argument("--early-stop-window", type=int, default=50)
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--query-budget", type=int, default=10_000)
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument(
        "--train-eval-k",
        type=int,
        default=2,
        help="Lower-cost EOT count used only by train-split environments.",
    )
    parser.add_argument("--support-scenes", type=int, default=4)
    parser.add_argument("--episode-steps", type=int, default=64)
    parser.add_argument("--max-prefix", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def _positive(value: int, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if int(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _parse_seeds(value: str) -> tuple[int, ...]:
    parts = str(value).split()
    if not parts:
        raise ValueError("--seeds must contain at least one non-negative integer")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("--seeds must contain only space-separated integers") from exc
    if any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain only non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must not contain duplicates")
    return seeds


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(output: Path, *, excluded: set[Path]) -> None:
    destination = output / "SHA256SUMS"
    excluded_resolved = {path.resolve() for path in excluded} | {destination.resolve()}
    records = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.resolve() not in excluded_resolved:
            records.append(f"{_sha256_file(path)}  {path.relative_to(output).as_posix()}")
    destination.write_text("\n".join(records) + "\n", encoding="utf-8")


def _output_has_experiment_content(output: Path) -> bool:
    """Return whether *output* contains anything owned by the matrix runner."""
    if not output.exists():
        return False
    if not output.is_dir():
        return True
    return any(
        entry.name not in EXTERNAL_LAUNCHER_ARTIFACTS
        for entry in output.iterdir()
    )


def _check_resume_compatibility(
    *,
    output: Path,
    config_sha256: str,
    selected_model_ids: Sequence[str],
    args: argparse.Namespace,
) -> None:
    provenance_path = output / "matrix_provenance.json"
    if not provenance_path.is_file():
        raise FileExistsError(
            f"--resume requires an existing matrix_provenance.json in {output}"
        )
    previous = _load_json(provenance_path)
    mismatches = []
    if previous.get("config_sha256") != config_sha256:
        mismatches.append("configuration content hash")
    if previous.get("selected_model_ids") != list(selected_model_ids):
        mismatches.append("selected model IDs/order")
    current_commit = _git_commit()
    if previous.get("git_commit") != current_commit:
        mismatches.append("git commit")
    previous_args = previous.get("arguments")
    experiment_keys = {
        "python_bin",
        "device",
        "seeds",
        "model_ids",
        "max_steps",
        "minimum_steps",
        "early_stop_success_rate",
        "early_stop_window",
        "save_freq",
        "episodes",
        "query_budget",
        "eval_k",
        "train_eval_k",
        "support_scenes",
        "episode_steps",
        "max_prefix",
    }
    if not isinstance(previous_args, dict):
        mismatches.append("recorded experiment arguments")
    else:
        changed = sorted(
            key
            for key in experiment_keys
            if previous_args.get(key) != getattr(args, key)
        )
        if changed:
            mismatches.append("experiment arguments: " + ", ".join(changed))
    if mismatches:
        raise ValueError(
            "refusing to mix incompatible artifacts under --resume; changed "
            + "; ".join(mismatches)
            + ". Choose a new --output directory."
        )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _completed(path: Path) -> bool:
    try:
        return path.read_text(encoding="utf-8").splitlines()[0].strip() == "COMPLETED"
    except (OSError, IndexError):
        return False


def _run_child(
    *,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    args: argparse.Namespace,
    output: Path,
) -> Dict[str, Any]:
    model_id = str(model["id"])
    model_output = output / "models" / model_id
    status_path = model_output / "STATUS.txt"
    if args.resume and _completed(status_path):
        return {
            "model_id": model_id,
            "detector": model["detector"],
            "detector_model": model["detector_model"],
            "requires_network": bool(model["requires_network"]),
            "status": "skipped_completed",
            "returncode": 0,
        }

    attack = config["attack"]
    patch = config["patch"]
    assets = config["assets"]
    environment = os.environ.copy()
    environment.update(
        {
            "RUN_ID": model_id,
            "RESULTS": str(model_output),
            "PYTHON_BIN": str(args.python_bin),
            "DEVICE": str(args.device),
            "SIM_SEEDS": str(args.seeds),
            "SIM_STEPS": str(args.max_steps),
            "SIM_MIN_STEPS": str(args.minimum_steps),
            "SIM_EARLY_STOP_RATE": str(args.early_stop_success_rate),
            "SIM_EARLY_STOP_WINDOW": str(args.early_stop_window),
            "SIM_SAVE_FREQ": str(args.save_freq),
            "SIM_EPISODES": str(args.episodes),
            "SIM_QUERY_BUDGET": str(args.query_budget),
            "SIM_EVAL_K": str(args.eval_k),
            "SIM_TRAIN_EVAL_K": str(args.train_eval_k),
            "SIM_SUPPORT_SCENES": str(args.support_scenes),
            "SIM_EPISODE_STEPS": str(args.episode_steps),
            "SIM_MAX_PREFIX": str(args.max_prefix),
            "SIM_GRID_CELL": str(patch["grid_cell"]),
            "SIM_AREA_CAP": str(patch["area_cap_fraction"]),
            "SIM_TRANSFORM_STRENGTH": str(patch["transform_strength"]),
            "SIM_PAINT": str(patch["paint"]),
            "SIM_PAINT_ACTION_MODE": str(patch["paint_action_mode"]),
            "SIM_PAINT_PALETTE": str(patch["paint_palette"]),
            "SIM_PHYSICS": str(patch["physics"]),
            "SIM_SIGN_IMAGE": str(assets["sign_day"]),
            "SIM_ACTIVE_IMAGE": str(assets["sign_active"]),
            "SIM_BGDIR": str(assets["backgrounds"]),
            "SIM_SOURCE_CLASS": str(attack["source_class"]),
            "SIM_ATTACK_MODE": str(attack["mode"]),
            "SIM_TARGET_CLASS": str(attack["target_class"] or ""),
            "SIM_ALLOWED_ALTS": str(attack["allowed_alternative_classes"] or ""),
            "SIM_DETECTOR": str(model["detector"]),
            "SIM_DETECTOR_MODEL": str(model["detector_model"]),
            "SIM_YOLO_VERSION": str(model["yolo_version"]),
            "SIM_WEIGHTS": str(model["weights"]),
            "RUN_TESTS": "0",
            "RUN_VISUALS": "0",
            "RESUME": "1" if args.resume else "0",
            "RESTART_PARTIAL": "1",
            # Avoid nesting nine child archives inside the matrix archive.
            "CREATE_ARCHIVE": "0",
        }
    )
    log_path = output / "driver_logs" / f"{model_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== matrix child attempt started {started} ===\n")
        log.flush()
        result = subprocess.run(
            ["bash", "tools/run_simulation_test.sh"],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "model_id": model_id,
        "detector": model["detector"],
        "detector_model": model["detector_model"],
        "requires_network": bool(model["requires_network"]),
        "started_utc": started,
        "finished_utc": _utc_now(),
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": int(result.returncode),
        "log": str(log_path.resolve()),
    }


def _aggregate(output: Path, models: Sequence[Mapping[str, Any]]) -> int:
    model_by_id = {str(model["id"]): model for model in models}
    rows: list[Dict[str, Any]] = []
    for model_id, model in model_by_id.items():
        source = output / "models" / model_id / "simulation_summary.csv"
        if not source.is_file():
            continue
        with source.open("r", newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                rows.append(
                    {
                        "model_id": model_id,
                        "detector": model["detector"],
                        "detector_model": model["detector_model"],
                        **raw,
                    }
                )
    destination = output / "matrix_summary.csv"
    fields = [
        "model_id",
        "detector",
        "detector_model",
        "family",
        "method",
        "seed",
        "status",
        "success_rate",
        "wilson_95_lower",
        "wilson_95_upper",
        "detector_queries",
        "offline_training_detector_queries",
        "development_prefix_detector_queries",
        "frozen_evaluation_detector_queries",
        "policy_environment_steps",
        "training_stop_reason",
        "evaluated_candidates",
        "best_joint_success",
        "best_score",
        "fidelity",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output / "matrix_summary.json",
        {
            "claim_scope": "SYNTHETIC_NON_EMPIRICAL_BLACK_BOX_SIMULATION_ONLY",
            "comparison_limit": (
                "RL query totals include offline training, development-prefix selection, "
                "and frozen evaluation. Baseline query totals are online search ledgers; "
                "baselines are not independently frozen-evaluated by this runner."
            ),
            "rows": rows,
        },
    )
    return len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    args.seeds = " ".join(str(seed) for seed in seeds)
    for name in (
        "max_steps",
        "episodes",
        "query_budget",
        "eval_k",
        "train_eval_k",
        "support_scenes",
        "episode_steps",
        "max_prefix",
        "early_stop_window",
        "save_freq",
    ):
        _positive(getattr(args, name), f"--{name.replace('_', '-')}")
    _positive(args.minimum_steps, "--minimum-steps", allow_zero=True)
    if args.minimum_steps > args.max_steps:
        raise ValueError("--minimum-steps cannot exceed --max-steps")
    if not 0.0 <= float(args.early_stop_success_rate) <= 1.0:
        raise ValueError("--early-stop-success-rate must be in [0, 1]")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config_path = config_path.resolve()
    config = validate_matrix(config_path)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()

    requested = {part.strip() for part in args.model_ids.split(",") if part.strip()}
    enabled = [model for model in config["models"] if model["enabled"]]
    known = {str(model["id"]) for model in enabled}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError("unknown or disabled --model-ids: " + ", ".join(unknown))
    models = [model for model in enabled if not requested or model["id"] in requested]
    if not models:
        raise ValueError("no enabled detector models were selected")

    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    output_has_content = _output_has_experiment_content(output)
    if output_has_content:
        if not args.resume:
            raise FileExistsError(
                f"output is not empty: {output}; use --resume only for an identical run "
                "or choose a new --output directory"
            )
        _check_resume_compatibility(
            output=output,
            config_sha256=config_sha256,
            selected_model_ids=[str(model["id"]) for model in models],
            args=args,
        )
        if _completed(output / "STATUS.txt") and all(
            _completed(output / "models" / str(model["id"]) / "STATUS.txt")
            for model in models
        ):
            print(f"[matrix] already completed: {output}", flush=True)
            return 0
    output.mkdir(parents=True, exist_ok=True)

    provenance = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "claim_scope": "SYNTHETIC_NON_EMPIRICAL_BLACK_BOX_SIMULATION_ONLY",
        "method_boundary": (
            "Detector checkpoints are frozen; each model gets a separately trained "
            "patch policy. Joint-palette runs optimize one paired DAY/active material "
            "choice per selected canonical cell."
        ),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "git_commit": _git_commit(),
        "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "selected_model_ids": [model["id"] for model in models],
        "arguments": vars(args),
        "declared_training_compute": {
            "detector_images_per_policy_step": int(
                2 * args.train_eval_k * args.support_scenes
            ),
            "maximum_detector_images_per_policy_excluding_episode_resets": int(
                args.max_steps * 2 * args.train_eval_k * args.support_scenes
            ),
            "maximum_matrix_detector_images_excluding_episode_resets": int(
                len(models)
                * len(seeds)
                * args.max_steps
                * 2
                * args.train_eval_k
                * args.support_scenes
            ),
            "note": (
                "Upper-bound planning estimate for overlay evaluations only; "
                "clean episode-reset references add queries, while the rolling "
                "success gate may stop before the maximum."
            ),
        },
    }
    estimate = provenance["declared_training_compute"][
        "maximum_matrix_detector_images_excluding_episode_resets"
    ]
    print(
        "[matrix] declared training overlay-query upper estimate: "
        f"{estimate:,} detector images (episode-reset references excluded)",
        flush=True,
    )
    _write_json(output / "matrix_provenance.json", provenance)
    _write_json(output / "resolved_matrix_config.json", config)
    (output / "STATUS.txt").write_text("RUNNING\n", encoding="utf-8")

    if not args.skip_tests:
        test_log = output / "driver_logs" / "pytest.log"
        test_log.parent.mkdir(parents=True, exist_ok=True)
        with test_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(args.python_bin), "-m", "pytest", "-q", "tests"],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            (output / "STATUS.txt").write_text("FAILED_TESTS\n", encoding="utf-8")
            return int(result.returncode)

    run_rows = []
    for model in models:
        print(f"[matrix] {model['id']} -> {output / 'driver_logs' / (str(model['id']) + '.log')}", flush=True)
        row = _run_child(model=model, config=config, args=args, output=output)
        run_rows.append(row)
        _write_json(output / "model_run_status.json", run_rows)
        if row["returncode"] != 0 and not args.continue_on_error:
            break

    summary_rows = _aggregate(output, models)
    failures = [row for row in run_rows if int(row["returncode"]) != 0]
    final_status = "COMPLETED" if not failures and len(run_rows) == len(models) else "FAILED"
    archive = output / "misclassification_matrix_results.tar.gz"
    try:
        (output / "STATUS.txt").write_text(
            f"{final_status}\nmodels_attempted={len(run_rows)}\n"
            f"summary_rows={summary_rows}\n",
            encoding="utf-8",
        )
        _write_checksums(output, excluded={archive})
        if not args.no_archive:
            with tarfile.open(archive, "w:gz") as handle:
                for path in sorted(output.rglob("*")):
                    if path.is_file() and path != archive:
                        handle.add(path, arcname=path.relative_to(output))
    except Exception:
        (output / "STATUS.txt").write_text(
            f"FAILED_FINALIZATION\nmodels_attempted={len(run_rows)}\n"
            f"summary_rows={summary_rows}\n",
            encoding="utf-8",
        )
        _write_checksums(output, excluded={archive})
        raise
    return 1 if failures or len(run_rows) != len(models) else 0


if __name__ == "__main__":
    raise SystemExit(main())
