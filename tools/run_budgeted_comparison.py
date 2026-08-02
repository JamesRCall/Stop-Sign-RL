"""Run native grid optimizers with identical query/material limits.

The environment JSON uses the same keys as the existing baseline scripts.  A
detector-image budget includes the fixed clean reference images acquired at
reset; candidate costs are fixed at ``2 * eval_K`` images.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.budgeted.protocol import BudgetSpec
from baselines.budgeted.json_io import (
    StrictJSONError,
    atomic_write_json_new,
    strict_json_load,
)
from baselines.budgeted.suite import NATIVE_RUNNERS, run_native_suite
from baselines.budgeted.traffic_sign_oracle import (
    TrafficSignCandidateOracle,
    exact_material_limit,
)
from baselines.grid_utils import (
    _default_cfg_for_env,
    build_env_from_args,
    resolve_yolo_weights,
)
from utils.experiment_manifest import package_versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budget-matched random/greedy/GA/ES/PSO/CMA-ES comparison"
    )
    parser.add_argument(
        "--environment-json",
        required=True,
        help="JSON object with baseline environment arguments.",
    )
    parser.add_argument(
        "--methods",
        default=(
            "random_search,forward_greedy,genetic_algorithm,gaussian_es,"
            "fipatch_style_pso_proxy,cma_es"
        ),
        help="Comma-separated native registry method IDs.",
    )
    parser.add_argument(
        "--detector-query-limit",
        type=int,
        required=True,
        help="Total detector input images, including clean references.",
    )
    material = parser.add_mutually_exclusive_group(required=True)
    material.add_argument("--material-pixel-limit", type=int)
    material.add_argument("--material-area-fraction", type=float)
    parser.add_argument("--scene-seed", type=int, default=123)
    parser.add_argument("--optimizer-seed", type=int, default=123)
    parser.add_argument(
        "--method-config-json",
        default="",
        help="Optional JSON mapping from method ID to keyword arguments.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load_object(path: str, label: str) -> Dict[str, Any]:
    try:
        value = strict_json_load(Path(path), label=label)
    except StrictJSONError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validate_environment_keys(config: Dict[str, Any]) -> None:
    allowed = set(_default_cfg_for_env({})) | {
        "action_indexing",
        "terminate_on_success",
        "seed",
    }
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            "unknown environment JSON keys: " + ", ".join(unknown)
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _experiment_artifact_identity(config: Dict[str, Any]) -> Dict[str, Any]:
    """Bind detector weights and the full background collection by content."""
    detector_type = str(config["detector"]).strip().lower()
    detector: Dict[str, Any] = {
        "type": detector_type,
        "model": str(config.get("detector_model", "")),
    }
    if detector_type == "yolo" and not str(config["detector_device"]).lower().startswith("server://"):
        weights = Path(
            resolve_yolo_weights(
                str(config["yolo_version"]), config.get("yolo_weights")
            )
        ).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"detector weights do not exist: {weights}")
        detector["weights"] = {
            "path": str(weights),
            "size_bytes": int(weights.stat().st_size),
            "sha256": _sha256_file(weights),
        }
    elif str(config["detector_device"]).lower().startswith("server://"):
        detector["remote_endpoint"] = str(config["detector_device"])
        detector["limitation"] = (
            "remote checkpoint bytes are not locally observable; server-side "
            "artifact identity must be recorded separately"
        )

    if str(config["bg_mode"]).lower() == "solid":
        backgrounds: Dict[str, Any] = {
            "mode": "synthetic_solid",
            "colors_rgb": [[200, 200, 200], [120, 120, 120], [30, 30, 30]],
        }
    else:
        folder = Path(str(config["bgdir"])).resolve()
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in supported
        ) if folder.is_dir() else []
        if not paths:
            raise FileNotFoundError(
                f"no supported background images for identity binding: {folder}"
            )
        backgrounds = {
            "mode": "dataset",
            "folder": str(folder),
            "files": [
                {
                    "name": path.name,
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
                for path in paths
            ],
        }
    return {"detector": detector, "background_collection": backgrounds}


def _runtime_identity() -> Dict[str, Any]:
    source_paths = (
        "tools/run_budgeted_comparison.py",
        "baselines/budgeted/protocol.py",
        "baselines/budgeted/optimizers.py",
        "baselines/budgeted/suite.py",
        "baselines/budgeted/traffic_sign_oracle.py",
        "envs/attack_objective.py",
        "envs/stop_sign_grid_env.py",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            (
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "ultralytics",
                "gymnasium",
                "cma",
            )
        ),
        "source_sha256": {
            relative: _sha256_file(Path(ROOT, relative))
            for relative in source_paths
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if os.path.lexists(output):
        raise FileExistsError(
            f"refusing to overwrite existing comparison report: {output}"
        )
    methods = tuple(
        part.strip() for part in str(args.methods).split(",") if part.strip()
    )
    if not methods:
        raise ValueError("methods must not be empty")
    unknown = [method for method in methods if method not in NATIVE_RUNNERS]
    if unknown:
        raise ValueError("unknown/non-native methods: " + ", ".join(unknown))
    raw_env_config = _load_object(args.environment_json, "environment JSON")
    _validate_environment_keys(raw_env_config)
    env_config = _default_cfg_for_env(raw_env_config)
    env_config["seed"] = int(args.scene_seed)
    artifact_identity = _experiment_artifact_identity(env_config)
    method_configs = (
        _load_object(args.method_config_json, "method config JSON")
        if args.method_config_json
        else {}
    )
    if any(not isinstance(value, dict) for value in method_configs.values()):
        raise ValueError("each method config must be a JSON object")

    def build_oracle() -> TrafficSignCandidateOracle:
        env = build_env_from_args(SimpleNamespace(**env_config))
        return TrafficSignCandidateOracle(
            env,
            scene_seed=int(args.scene_seed),
            eval_k=int(env_config["eval_K"]),
            experiment_artifact_identity=artifact_identity,
        )

    # Reuse this fully accounted oracle for the first method instead of making
    # an unreported probe solely to discover exact sign-pixel area.
    first_oracle = build_oracle()
    cached = [first_oracle]
    try:
        if args.material_pixel_limit is not None:
            material_limit = int(args.material_pixel_limit)
        else:
            material_limit = exact_material_limit(
                first_oracle.sign_material_pixels,
                float(args.material_area_fraction),
            )
        if (
            first_oracle.objective_material_pixel_limit is not None
            and material_limit
            != first_oracle.objective_material_pixel_limit
        ):
            raise ValueError(
                "requested material limit does not match env.area_cap_frac: "
                f"requested={material_limit}, objective="
                f"{first_oracle.objective_material_pixel_limit}; set both "
                "budgets to the same exact sign-pixel limit"
            )

        def oracle_factory() -> TrafficSignCandidateOracle:
            return cached.pop() if cached else build_oracle()

        budget = BudgetSpec(
            detector_query_limit=int(args.detector_query_limit),
            material_pixel_limit=material_limit,
        )
        report = run_native_suite(
            oracle_factory,
            budget=budget,
            methods=methods,
            seed=int(args.optimizer_seed),
            method_configs=method_configs,
        )
        payload = report.to_dict()
        payload["invocation"] = {
            "environment_json": str(Path(args.environment_json).resolve()),
            "environment_json_sha256": _sha256_file(
                Path(args.environment_json).resolve()
            ),
            "method_config_json": (
                str(Path(args.method_config_json).resolve())
                if args.method_config_json
                else None
            ),
            "method_config_json_sha256": (
                _sha256_file(Path(args.method_config_json).resolve())
                if args.method_config_json
                else None
            ),
            "environment_config": env_config,
            "experiment_artifact_identity": artifact_identity,
            "runtime_identity": _runtime_identity(),
            "methods": list(methods),
            "scene_seed": int(args.scene_seed),
            "optimizer_seed": int(args.optimizer_seed),
            "budget": asdict(budget),
            "material_area_fraction_requested": args.material_area_fraction,
        }
        atomic_write_json_new(output, payload)
    finally:
        for unused_oracle in cached:
            unused_oracle.close()
    print(f"Saved budget-matched comparison to {output}")


if __name__ == "__main__":
    main()
