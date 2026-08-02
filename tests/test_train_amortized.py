from types import SimpleNamespace

from PIL import Image
import pytest

import train_amortized
from detectors.class_names import resolve_class_id
from envs.traffic_sign_grid_env import TrafficSignGridEnv
from train_amortized import (
    RollingSuccessGate,
    _detector_class_binding,
    _transport_paint,
    _validate_calibration,
    _validate_research_splits,
    build_amortized_environment,
)
from utils.task_manifest import TaskManifest, TaskSpec, load_task_manifest


def test_rolling_success_gate_waits_for_full_window_and_minimum_steps():
    gate = RollingSuccessGate(success_rate=0.75, window=4, minimum_steps=100)
    assert not gate.observe(success=True, policy_steps=25)
    assert not gate.observe(success=True, policy_steps=50)
    assert not gate.observe(success=False, policy_steps=75)
    assert not gate.observe(success=True, policy_steps=99)
    assert gate.observe(success=True, policy_steps=100)
    report = gate.report(final_policy_steps=100)
    assert report["triggered"] is True
    assert report["trigger_policy_step"] == 100
    assert report["final_rolling_success_rate"] == pytest.approx(0.75)
    assert report["stop_reason"] == "rolling_joint_success_gate"


def test_zero_rate_disables_rolling_success_gate_but_records_history():
    gate = RollingSuccessGate(success_rate=0.0, window=2, minimum_steps=0)
    assert not gate.observe(success=True, policy_steps=1)
    assert not gate.observe(success=True, policy_steps=2)
    report = gate.report(final_policy_steps=10)
    assert report["enabled"] is False
    assert report["triggered"] is False
    assert report["stop_reason"] == "maximum_total_steps"


def test_template_has_all_research_splits_and_is_explicitly_non_empirical():
    manifest = load_task_manifest("configs/amortized_tasks.template.json")
    args = SimpleNamespace(
        allow_single_task_debug=False,
        allow_incomplete_splits=False,
    )
    _validate_research_splits(manifest, args)
    task = manifest.tasks_for_split("train")[0]
    with pytest.raises(ValueError, match="non-empirical"):
        _validate_calibration(manifest, task, allow_uncalibrated=False)
    calibration = _validate_calibration(manifest, task, allow_uncalibrated=True)
    assert calibration is not None
    assert calibration.provenance.is_empirical is False


def test_spectral_transport_becomes_a_seeded_effective_coated_cell_color():
    manifest = load_task_manifest("configs/amortized_tasks.template.json")
    task = manifest.tasks_for_split("train")[0]
    calibration = _validate_calibration(manifest, task, allow_uncalibrated=True)
    first_paint, first_result = _transport_paint(calibration, seed=123)
    second_paint, second_result = _transport_paint(calibration, seed=123)
    assert first_paint == second_paint
    assert first_result.to_dict() == second_result.to_dict()
    assert first_paint.day_alpha == 1.0
    assert first_paint.active_alpha == 1.0
    assert first_paint.day_hex != first_paint.active_hex


class _SharedClassMapDetector:
    id_to_name = {1: "speed limit 25", 2: "speed limit 35"}

    def __init__(self, initial_source):
        self.target_id = self.resolve_class_id(initial_source, role="source class")

    def resolve_class_id(self, class_ref, *, role="class"):
        return resolve_class_id(self.id_to_name, class_ref, role=role)


def _task(task_id: str, source_class: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        split="train",
        sign_instance_id=f"sign-{task_id}",
        background_split_id=f"background-{task_id}",
        camera_id=f"camera-{task_id}",
        material_batch_id=f"material-{task_id}",
        physical_run_group=f"run-{task_id}",
        detector_id="shared-fine-grained-detector",
        source_class=source_class,
        target_class=None,
        attack_mode="disappearance",
        calibration_sha256="",
        condition_features=(0.0,),
        weight=1.0,
        environment={},
    )


def test_shared_detector_resolves_each_task_source_class_independently(
    monkeypatch, tmp_path
):
    manifest = TaskManifest(
        manifest_id="shared-detector-source-binding",
        description="unit-test manifest",
        condition_feature_names=("variant",),
        leakage_keys=(),
        tasks=(
            _task("speed-25", "speed limit 25"),
            _task("speed-35", "speed limit 35"),
        ),
        canonical_sha256="1" * 64,
        source_sha256="2" * 64,
        source_path=str(tmp_path / "manifest.json"),
    )
    args = SimpleNamespace(
        support_scenes=1,
        seed=17,
        allow_uncalibrated_simulation=True,
        risk_alpha=0.25,
        required_support_success_rate=0.8,
        required_clean_eligible_rate=0.8,
        required_day_preservation_rate=0.8,
        dual_learning_rate=0.05,
    )
    created_detectors = []

    def fake_build_env(config):
        detector = config.detector_instance
        if detector is None:
            detector = _SharedClassMapDetector(config.source_class)
            created_detectors.append(detector)
        sign = Image.new("RGBA", (32, 32), (220, 220, 220, 255))
        return TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=sign,
            background_images=[Image.new("RGB", (64, 64), (128, 128, 128))],
            pole_image=None,
            detector_instance=detector,
            source_class=config.source_class,
            attack_mode=config.attack_mode,
            steps_per_episode=2,
            eval_K=1,
            grid_cell_px=8,
            action_indexing=config.action_indexing,
            terminate_on_success=bool(config.terminate_on_success),
            img_size=(64, 64),
            obs_size=(32, 32),
            seed=config.seed,
        )

    monkeypatch.setattr(
        train_amortized,
        "_validate_calibration",
        lambda *unused_args, **unused_kwargs: None,
    )
    monkeypatch.setattr(train_amortized, "build_env_from_args", fake_build_env)

    environment = build_amortized_environment(manifest, args, split="train")
    first = environment.tasks[0].support_envs[0].unwrapped
    second = environment.tasks[1].support_envs[0].unwrapped

    assert len(created_detectors) == 1
    assert first.det is second.det is created_detectors[0]
    assert first.det.target_id == 1  # legacy wrapper binding remains unchanged
    assert first.source_class_id == 1
    assert second.source_class_id == 2

    second_binding = _detector_class_binding(second)
    assert second_binding["source_class_id"] == 2
    assert second_binding["source_class_name"] == "speed limit 35"
    assert second_binding["legacy_wrapper_target_id"] == 1
    assert second_binding["label_map_class_count"] == 2
    assert len(second_binding["label_map_sha256"]) == 64
