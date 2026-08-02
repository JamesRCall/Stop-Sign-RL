"""Freeze a finite family of task-conditioned development prefixes.

The generator intentionally permits only the development split in a
paper-facing run.  It never reads calibration or certification outcomes.  The
result is a content-addressed candidate-family artifact for the separate risk
selection protocol; it is not itself a success certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_amortized import build_amortized_environment
from utils.experiment_manifest import write_manifest
from utils.task_manifest import TaskManifest, TaskSpec, load_task_manifest


SCHEMA_VERSION = 1
METHOD = "task_amortized_prefix_valid_support_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and hash every prefix of a trained task-conditioned "
            "policy on development tasks only."
        )
    )
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--max-prefix-length", type=int, default=0)
    parser.add_argument(
        "--max-development-detector-queries-per-task",
        type=int,
        default=0,
        help=(
            "Hard detector-image budget per task, including reset/baseline "
            "queries; 0 means no additional cap."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-nonempirical-debug",
        action="store_true",
        help="Debug only: permit a training run marked as uncalibrated simulation.",
    )
    parser.add_argument(
        "--allow-incomplete-training-ledger-debug",
        action="store_true",
        help="Debug only: permit a run manifest without final training-query totals.",
    )
    return parser.parse_args()


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def artifact_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    return canonical_sha256(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact(path_value: str, *, suffix: str = "") -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        return path
    if suffix and not str(path).lower().endswith(suffix.lower()):
        candidate = Path(str(path) + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _resolved_file_hash(config: Mapping[str, Any], key: str) -> str | None:
    raw = config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    return _sha256_file(path)


def _task_seed(root_seed: int, task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return int((int(root_seed) + offset) % (2**31 - 1))


def _vectorize_observation(
    observation: Mapping[str, np.ndarray], normalizer: Any
) -> Dict[str, np.ndarray]:
    """Reproduce VecTransposeImage then VecNormalize for one raw observation."""

    result: Dict[str, np.ndarray] = {}
    for key, value in observation.items():
        array = np.asarray(value)
        if key == "image":
            if array.ndim != 3:
                raise ValueError("image observation must have HWC rank 3")
            array = np.transpose(array, (2, 0, 1))
        result[key] = np.expand_dims(array, axis=0)
    normalized = normalizer.normalize_obs(result)
    return {key: np.asarray(value) for key, value in normalized.items()}


def _pattern_descriptor(
    *,
    manifest: TaskManifest,
    task: TaskSpec,
    grid_shape: tuple[int, int],
    prefix_info: Mapping[str, Any],
) -> Dict[str, Any]:
    config = task.resolved_environment(manifest.directory)
    selected_pixels = int(prefix_info["selected_pixels"])
    sign_pixels = int(prefix_info["sign_pixels"])
    if selected_pixels <= 0 or sign_pixels <= 0 or selected_pixels > sign_pixels:
        raise ValueError(f"task {task.task_id!r} returned invalid exact area")
    actions = [int(value) for value in prefix_info["ordered_actions"]]
    selected = [int(value) for value in prefix_info["selected_indices"]]
    if len(actions) != len(selected) or sorted(actions) != selected:
        raise ValueError(f"task {task.task_id!r} returned an invalid prefix sequence")
    return {
        "schema_version": 1,
        "coordinate_system": "canonical_full_grid_row_major",
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "task_id": task.task_id,
        "source_class": task.source_class,
        "target_class": task.target_class,
        "attack_mode": task.attack_mode,
        "sign_instance_id": task.sign_instance_id,
        "material_batch_id": task.material_batch_id,
        "calibration_sha256": task.calibration_sha256,
        "sign_day_asset_sha256": _resolved_file_hash(config, "sign_image"),
        "sign_active_asset_sha256": _resolved_file_hash(config, "sign_active_image"),
        "ordered_actions": actions,
        "selected_indices": selected,
        "selected_pixels": selected_pixels,
        "sign_pixels": sign_pixels,
    }


def assemble_candidate_family(
    task_records: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Align task-specific sequences by prefix order without assuming success monotonicity."""

    if not task_records:
        raise ValueError("candidate family requires at least one task record")
    lengths = [len(record.get("prefixes", [])) for record in task_records]
    common_length = min(lengths)
    if common_length < 1:
        raise ValueError("every development task must produce at least one prefix")
    candidates: list[Dict[str, Any]] = []
    for order in range(1, common_length + 1):
        patterns = []
        for record in task_records:
            prefix = record["prefixes"][order - 1]
            if int(prefix["order"]) != order:
                raise ValueError("task prefix records must be contiguous and one-indexed")
            patterns.append(
                {
                    "task_id": str(record["task_id"]),
                    "pattern_sha256": str(prefix["pattern_sha256"]),
                    "selected_pixels": int(prefix["selected_pixels"]),
                    "sign_pixels": int(prefix["sign_pixels"]),
                }
            )
        candidates.append(
            {
                "prefix_id": f"prefix-{order:04d}",
                "order": order,
                "task_patterns": patterns,
            }
        )
    return candidates


def _validated_run_configuration(
    run_record: Mapping[str, Any],
    manifest: TaskManifest,
    *,
    allow_nonempirical_debug: bool,
    allow_incomplete_ledger_debug: bool,
) -> Dict[str, Any]:
    if run_record.get("method") != METHOD:
        raise ValueError("run manifest method does not match the amortized policy")
    manifest_record = run_record.get("task_manifest")
    if not isinstance(manifest_record, dict):
        raise ValueError("run manifest lacks task_manifest metadata")
    if manifest_record.get("canonical_sha256") != manifest.canonical_sha256:
        raise ValueError("task manifest canonical hash differs from the training run")
    if manifest_record.get("source_sha256") != manifest.source_sha256:
        raise ValueError("task manifest source hash differs from the training run")
    limits = run_record.get("claim_limits")
    if not isinstance(limits, dict):
        raise ValueError("run manifest lacks claim_limits")
    nonempirical = bool(limits.get("uncalibrated_simulation", False))
    if nonempirical and not allow_nonempirical_debug:
        raise ValueError(
            "training run is marked uncalibrated simulation; use the debug override "
            "only for software testing"
        )
    accounting = run_record.get("query_accounting")
    complete = isinstance(accounting, dict) and isinstance(
        accounting.get("offline_training_detector_image_queries"), int
    )
    completed_run = run_record.get("run_status") == "completed"
    if (not complete or not completed_run) and not allow_incomplete_ledger_debug:
        raise ValueError("training run lacks finalized detector-image query accounting")
    config = run_record.get("config")
    if not isinstance(config, dict):
        raise ValueError("run manifest lacks its training config")
    result = dict(config)
    required = (
        "support_scenes",
        "risk_alpha",
        "required_support_success_rate",
        "required_clean_eligible_rate",
        "required_day_preservation_rate",
        "dual_learning_rate",
        "seed",
        "allow_uncalibrated_simulation",
    )
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"run manifest config is incomplete: missing {missing}")
    final_duals = run_record.get("final_constraint_duals")
    if not isinstance(final_duals, dict):
        if not allow_incomplete_ledger_debug:
            raise ValueError("training run lacks finalized constraint duals")
        final_duals = {"failure": 0.0, "day": 0.0, "clean": 0.0, "area": 0.0}
    result["initial_constraint_duals"] = dict(final_duals)
    # Dual adaptation is an offline training mechanism.  A frozen generator
    # must not change behavior based on the order in which development tasks
    # happen to be exported.
    result["dual_learning_rate"] = 0.0
    result["terminate_on_support_success"] = False
    return result


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.max_prefix_length) < 0:
        raise ValueError("max-prefix-length must be >= 0")
    if int(args.max_development_detector_queries_per_task) < 0:
        raise ValueError("development detector-query budget must be >= 0")
    manifest = load_task_manifest(args.task_manifest)
    development_tasks = manifest.tasks_for_split("development")
    if not development_tasks:
        raise ValueError("task manifest contains no development tasks")

    run_path = _resolve_artifact(args.run_manifest)
    run_record = _load_json(run_path)
    config = _validated_run_configuration(
        run_record,
        manifest,
        allow_nonempirical_debug=bool(args.allow_nonempirical_debug),
        allow_incomplete_ledger_debug=bool(
            args.allow_incomplete_training_ledger_debug
        ),
    )
    root_seed = int(config["seed"] if args.seed is None else args.seed)
    config["seed"] = root_seed
    model_path = _resolve_artifact(args.model, suffix=".zip")
    normalizer_path = _resolve_artifact(args.vecnormalize)
    actual_policy_hash = _sha256_file(model_path)
    actual_normalizer_hash = _sha256_file(normalizer_path)
    sealed = run_record.get("final_artifacts")
    if isinstance(sealed, dict):
        if sealed.get("policy_sha256") != actual_policy_hash:
            raise ValueError("policy hash differs from the sealed training artifact")
        if sealed.get("vecnormalize_sha256") != actual_normalizer_hash:
            raise ValueError("VecNormalize hash differs from the sealed training artifact")
    elif not args.allow_incomplete_training_ledger_debug:
        raise ValueError("training run does not seal its final policy artifacts")

    raw_env = build_amortized_environment(
        manifest,
        SimpleNamespace(**config),
        split="development",
    )
    task_records: list[Dict[str, Any]] = []
    forward_calls = 0
    try:
        # Local imports keep artifact/schema inspection usable without the RL stack.
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecNormalize,
            VecTransposeImage,
        )

        wrapped = ActionMasker(
            raw_env, lambda value: value.unwrapped.action_masks()
        )
        dummy = DummyVecEnv([lambda: wrapped])
        transposed = VecTransposeImage(dummy)
        normalizer = VecNormalize.load(str(normalizer_path), transposed)
        normalizer.training = False
        normalizer.norm_reward = False
        model = MaskablePPO.load(
            str(model_path),
            env=None,
            device=str(args.device),
        )

        for task in development_tasks:
            seed = _task_seed(root_seed, task.task_id)
            observation, reset_info = raw_env.reset(
                seed=seed, options={"task_id": task.task_id}
            )
            prefixes: list[Dict[str, Any]] = []
            final_info: Mapping[str, Any] = reset_info
            terminated = False
            truncated = False
            query_budget_exhausted = False
            while True:
                if args.max_prefix_length and len(prefixes) >= int(
                    args.max_prefix_length
                ):
                    break
                mask = np.asarray(raw_env.action_masks(), dtype=bool)
                if not np.any(mask):
                    break
                query_budget = int(args.max_development_detector_queries_per_task)
                projected_queries = int(raw_env._episode_queries) + int(
                    raw_env.next_step_detector_query_cost()
                )
                if query_budget and projected_queries > query_budget:
                    query_budget_exhausted = True
                    break
                policy_observation = _vectorize_observation(observation, normalizer)
                action, _ = model.predict(
                    policy_observation,
                    deterministic=True,
                    action_masks=np.expand_dims(mask, axis=0),
                )
                action_index = int(np.asarray(action).reshape(-1)[0])
                if not bool(mask[action_index]):
                    raise RuntimeError("policy returned an action excluded by its mask")
                observation, _, terminated, truncated, info = raw_env.step(action_index)
                forward_calls += 1
                final_info = info
                descriptor = _pattern_descriptor(
                    manifest=manifest,
                    task=task,
                    grid_shape=(raw_env.Gh, raw_env.Gw),
                    prefix_info=info,
                )
                prefixes.append(
                    {
                        "order": len(prefixes) + 1,
                        "ordered_actions": list(descriptor["ordered_actions"]),
                        "selected_indices": list(descriptor["selected_indices"]),
                        "selected_pixels": int(descriptor["selected_pixels"]),
                        "sign_pixels": int(descriptor["sign_pixels"]),
                        "exact_area_fraction": dict(info["exact_area_fraction"]),
                        "pattern_descriptor": descriptor,
                        "pattern_sha256": canonical_sha256(descriptor),
                        "support_measurements": {
                            "joint_success": bool(info["attack_success"]),
                            "joint_success_rate": float(
                                info["support_joint_success_rate"]
                            ),
                            "objective_success_rate": float(
                                info["support_objective_success_rate"]
                            ),
                            "clean_eligible_rate": float(
                                info["support_clean_eligible_rate"]
                            ),
                            "day_preservation_rate": float(
                                info["support_day_preservation_rate"]
                            ),
                            "risk": dict(info["risk"]),
                        },
                        "detector_image_queries_episode": int(
                            info["support_detector_queries_episode"]
                        ),
                    }
                )
                if terminated or truncated:
                    break

            episode_queries = int(
                final_info.get(
                    "support_detector_queries_episode", raw_env._episode_queries
                )
            )
            task_records.append(
                {
                    "task_id": task.task_id,
                    "task_split": task.split,
                    "task_seed": seed,
                    "source_class": task.source_class,
                    "target_class": task.target_class,
                    "attack_mode": task.attack_mode,
                    "sequence_length": len(prefixes),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "query_budget_exhausted": query_budget_exhausted,
                    "detector_image_queries": episode_queries,
                    "prefixes": prefixes,
                }
            )
    finally:
        # VecNormalize owns the wrapped raw environment after construction.  A
        # build/load failure can happen before it exists, hence the fallback.
        if "normalizer" in locals():
            normalizer.close()
        else:
            raw_env.close()

    candidates = assemble_candidate_family(task_records)
    training_queries = None
    if isinstance(run_record.get("query_accounting"), dict):
        training_queries = run_record["query_accounting"].get(
            "offline_training_detector_image_queries"
        )
    output: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_candidate_family_not_a_certificate",
        "claim_scope": (
            "development-generated finite prefix family; calibration and "
            "certification outcomes were not read"
        ),
        "source_artifacts": {
            "task_manifest_source_sha256": manifest.source_sha256,
            "task_manifest_canonical_sha256": manifest.canonical_sha256,
            "run_manifest_sha256": _sha256_file(run_path),
            "policy_sha256": actual_policy_hash,
            "vecnormalize_sha256": actual_normalizer_hash,
        },
        "generation": {
            "split": "development",
            "root_seed": root_seed,
            "deterministic_policy": True,
            "support_scenes_per_task": int(config["support_scenes"]),
            "maximum_requested_prefix_length": int(args.max_prefix_length),
            "maximum_development_detector_queries_per_task": int(
                args.max_development_detector_queries_per_task
            ),
            "common_candidate_count": len(candidates),
        },
        "query_accounting": {
            "offline_training_detector_image_queries": training_queries,
            "development_prefix_detector_image_queries": sum(
                int(record["detector_image_queries"]) for record in task_records
            ),
            "policy_forward_calls": forward_calls,
            "calibration_detector_image_queries": 0,
            "certification_detector_image_queries": 0,
        },
        "tasks": task_records,
        "candidate_prefixes": candidates,
        "limitations": {
            "novelty_or_acceptance_guaranteed": False,
            "success_certified_by_this_artifact": False,
            "nonempirical_debug": bool(args.allow_nonempirical_debug),
            "incomplete_training_ledger_debug": bool(
                args.allow_incomplete_training_ledger_debug
            ),
        },
    }
    output["artifact_sha256"] = artifact_sha256(output)
    return output


def main() -> None:
    args = parse_args()
    output = generate(args)
    destination = Path(args.out_json).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(destination, output)
    print(f"wrote {destination}")
    print(f"artifact_sha256={output['artifact_sha256']}")


if __name__ == "__main__":
    main()
