from __future__ import annotations

import hashlib
import json

from PIL import Image, ImageDraw
import pytest

from baselines.budgeted.external import (
    EXTERNAL_RUNNER_API,
    ExternalArtifactManifest,
    ExternalMethodUnavailable,
    run_external_adapter,
)
from baselines.budgeted.json_io import (
    StrictJSONError,
    atomic_write_json_new,
    strict_json_dumps,
    strict_json_loads,
)
from baselines.budgeted.optimizers import (
    cma_es_search,
    evolution_strategy_search,
    fipatch_style_pso_proxy_search,
    genetic_search,
    greedy_search,
    random_search,
)
from baselines.budgeted.protocol import (
    BudgetExhausted,
    BudgetSpec,
    BudgetedEvaluator,
    MaterialBudgetExceeded,
    OracleObservation,
)
from baselines.budgeted.suite import registry_snapshot, run_native_suite
from baselines.budgeted.traffic_sign_oracle import (
    TrafficSignCandidateOracle,
    exact_material_limit,
)
from detectors.class_names import resolve_class_id
from envs.stop_sign_grid_env import TrafficSignGridEnv
from baselines.grid_utils import _default_cfg_for_env
from tools.run_budgeted_comparison import (
    _experiment_artifact_identity,
    _load_object,
    _validate_environment_keys,
)


class StubOracle:
    dimension = 5
    selectable_indices = (0, 1, 2, 3, 4)
    cell_material_pixels = (2, 3, 5, 7, 11)
    sign_material_pixels = 30
    detector_queries_per_evaluation = 2
    initial_detector_queries = 4
    objective_id = "stub-objective-v1"

    def __init__(self, objective_id="stub-objective-v1"):
        self.objective_id = objective_id
        self.detector_queries_total = self.initial_detector_queries
        self.closed = False

    def evaluate(self, selected_indices):
        self.detector_queries_total += self.detector_queries_per_evaluation
        score = sum((index + 1) for index in selected_indices) - 0.01 * sum(
            self.cell_material_pixels[index] for index in selected_indices
        )
        return OracleObservation(
            score=score,
            joint_success=(4 in selected_indices),
            metrics={"cardinality": len(selected_indices)},
        )

    def close(self):
        self.closed = True


def _evaluator(query_limit=24, material_limit=12):
    return BudgetedEvaluator(
        StubOracle(),
        BudgetSpec(
            detector_query_limit=query_limit,
            material_pixel_limit=material_limit,
        ),
    )


def test_budget_guard_rejects_before_detector_oracle_call():
    evaluator = _evaluator(query_limit=9, material_limit=10)
    assert (
        evaluator.budget.detector_query_scope
        == "online_search_including_clean_references"
    )
    first = evaluator.evaluate((0, 1))
    second = evaluator.evaluate((2,))
    assert first.selected_material_pixels == 5
    assert first.material_fraction == pytest.approx(5 / 30)
    assert second.detector_queries_cumulative == 8

    counter = evaluator.oracle.detector_queries_total
    with pytest.raises(BudgetExhausted, match="only 1 remain"):
        evaluator.evaluate((0,))
    assert evaluator.oracle.detector_queries_total == counter

    with pytest.raises(MaterialBudgetExceeded, match="limit is 10"):
        evaluator.evaluate((4,))
    assert evaluator.oracle.detector_queries_total == counter


def test_result_distinguishes_scalar_winner_from_best_joint_success():
    evaluator = _evaluator(query_limit=12, material_limit=12)
    scalar_winner = evaluator.evaluate((0, 1, 2))
    successful = evaluator.evaluate((4,))
    assert scalar_winner.score > successful.score
    result = evaluator.result(
        method="test",
        seed=0,
        implementation="test",
    )
    assert result.best == scalar_winner
    assert result.best_successful == successful
    payload = result.to_dict()
    assert payload["best"]["selected_indices"] == [0, 1, 2]
    assert payload["best_successful"]["selected_indices"] == [4]

def test_guard_rejects_duplicates_nonselectable_and_invalid_contracts():
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="duplicates"):
        evaluator.evaluate((1, 1))
    with pytest.raises(ValueError, match="out-of-range"):
        evaluator.evaluate((8,))

    oracle = StubOracle()
    oracle.selectable_indices = (0, 2, 4)
    restricted = BudgetedEvaluator(
        oracle, BudgetSpec(detector_query_limit=10, material_pixel_limit=12)
    )
    with pytest.raises(ValueError, match="non-selectable"):
        restricted.evaluate((1,))


@pytest.mark.parametrize(
    "runner,kwargs,method,fidelity",
    [
        (random_search, {}, "random_search", "native_implementation"),
        (greedy_search, {}, "forward_greedy", "native_implementation"),
        (
            genetic_search,
            {"population_size": 4},
            "genetic_algorithm",
            "native_implementation",
        ),
        (
            evolution_strategy_search,
            {"population_size": 4},
            "gaussian_es",
            "native_implementation",
        ),
        (
            fipatch_style_pso_proxy_search,
            {"swarm_size": 4},
            "fipatch_style_pso_proxy",
            "proxy_not_fipatch_reproduction",
        ),
    ],
)
def test_native_optimizers_are_deterministic_and_budget_feasible(
    runner, kwargs, method, fidelity
):
    result_a = runner(_evaluator(), seed=91, **kwargs)
    result_b = runner(_evaluator(), seed=91, **kwargs)
    assert result_a.method == method
    assert result_a.fidelity == fidelity
    assert result_a.detector_queries_used <= result_a.budget.detector_query_limit
    assert result_a.evaluated_candidates == len(result_a.trace)
    assert result_a.evaluated_candidates > 0
    assert all(
        row.selected_material_pixels <= result_a.budget.material_pixel_limit
        for row in result_a.trace
    )
    assert [row.to_dict() for row in result_a.trace] == [
        row.to_dict() for row in result_b.trace
    ]


def test_optional_cma_es_fails_truthfully_when_package_is_absent(monkeypatch):
    from baselines.budgeted import optimizers

    real_import = optimizers.importlib.import_module

    def reject_cma(name):
        if name == "cma":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(optimizers.importlib, "import_module", reject_cma)
    evaluator = _evaluator()
    result = cma_es_search(evaluator, seed=1)
    assert result.status == "unavailable"
    assert result.fidelity == "reference_package"
    assert result.evaluated_candidates == 0
    assert result.detector_queries_used == StubOracle.initial_detector_queries
    assert "declared 'cma' dependency" in result.note


def test_cma_reference_package_path_obeys_common_budget(monkeypatch):
    from baselines.budgeted import optimizers

    strategies = []

    class FakeStrategy:
        def __init__(self, mean, sigma, options):
            assert len(mean) == len(StubOracle.selectable_indices)
            assert sigma == 1.0
            self.population_size = options["popsize"]
            self.tell_calls = 0
            strategies.append(self)

        def ask(self):
            return [
                [1.0 if position == member else -1.0 for position in range(5)]
                for member in range(self.population_size)
            ]

        def tell(self, points, losses):
            assert len(points) == self.population_size
            assert len(losses) == self.population_size
            self.tell_calls += 1

    fake_cma = type("FakeCMA", (), {"CMAEvolutionStrategy": FakeStrategy})
    real_import = optimizers.importlib.import_module
    monkeypatch.setattr(
        optimizers.importlib,
        "import_module",
        lambda name: fake_cma if name == "cma" else real_import(name),
    )
    monkeypatch.setattr(optimizers.importlib_metadata, "version", lambda name: "4.4.4")

    result = cma_es_search(
        _evaluator(query_limit=12), seed=4, population_size=2
    )
    assert result.status == "query_budget_exhausted"
    assert result.implementation == "cma-package-4.4.4"
    assert result.evaluated_candidates == 4
    assert strategies[0].tell_calls == 2


def _write_external_manifest(tmp_path, method_id="patchattack"):
    artifact = tmp_path / "upstream_runner.py"
    artifact.write_text("# pinned upstream wrapper\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    mapping = tmp_path / "candidate_mapping.json"
    mapping.write_text('{"mapping":"upstream-to-binary-grid"}\n', encoding="utf-8")
    mapping_digest = hashlib.sha256(mapping.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "method_id": method_id,
        "paper_url": "https://example.invalid/paper",
        "upstream_repository": "https://example.invalid/repository",
        "upstream_revision": "0123456789abcdef",
        "artifact_path": artifact.name,
        "artifact_sha256": digest,
        "candidate_mapping_path": mapping.name,
        "candidate_mapping_sha256": mapping_digest,
        "candidate_mapping_status": "declared_preregistered_and_audited",
        "offline_detector_queries": 0,
        "runner_api": EXTERNAL_RUNNER_API,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, digest, mapping_digest


class StubExternalRunner:
    method_id = "patchattack"
    runner_api = EXTERNAL_RUNNER_API
    query_access = "budgeted_evaluator_only"

    def __init__(self, digest, mapping_digest):
        self.artifact_sha256 = digest
        self.candidate_mapping_sha256 = mapping_digest
        self.candidate_mapping_status = "declared_preregistered_and_audited"
        self.calls = 0

    def initialize(self, context):
        assert context["method_id"] == self.method_id

    def propose(self, history):
        proposals = [(0,), (0, 1), None]
        proposal = proposals[self.calls]
        self.calls += 1
        return proposal


def test_external_adapter_requires_exact_artifact_and_runner(tmp_path):
    manifest, digest, mapping_digest = _write_external_manifest(tmp_path)
    evaluator = _evaluator()
    before = evaluator.oracle.detector_queries_total
    with pytest.raises(ExternalMethodUnavailable, match="exact interactive"):
        run_external_adapter(
            evaluator,
            method_id="patchattack",
            seed=3,
            artifact_manifest_path=str(manifest),
            runner=None,
        )
    assert evaluator.oracle.detector_queries_total == before

    runner = StubExternalRunner(digest, mapping_digest)
    result = run_external_adapter(
        _evaluator(),
        method_id="patchattack",
        seed=3,
        artifact_manifest_path=str(manifest),
        runner=runner,
    )
    assert result.fidelity == "verified_external_adapter_with_declared_mapping"
    assert "not semantic equivalence" in result.note
    assert result.evaluated_candidates == 2
    assert result.trace[-1].selected_indices == (0, 1)
    assert result.implementation.endswith("@0123456789abcdef")


def test_external_manifest_hash_mismatch_fails_before_queries(tmp_path):
    manifest, digest, mapping_digest = _write_external_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    evaluator = _evaluator()
    before = evaluator.oracle.detector_queries_total
    with pytest.raises(ExternalMethodUnavailable, match="SHA-256 mismatch"):
        run_external_adapter(
            evaluator,
            method_id="patchattack",
            seed=3,
            artifact_manifest_path=str(manifest),
            runner=StubExternalRunner(digest, mapping_digest),
        )
    assert evaluator.oracle.detector_queries_total == before


def test_external_manifest_strict_json_and_mapping_hash_fail_closed(tmp_path):
    manifest, digest, mapping_digest = _write_external_manifest(tmp_path)
    manifest.write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )
    with pytest.raises(ExternalMethodUnavailable, match="duplicate JSON object key"):
        ExternalArtifactManifest.load(
            str(manifest), expected_method_id="patchattack"
        )

    manifest, digest, mapping_digest = _write_external_manifest(tmp_path)
    mapping = tmp_path / "candidate_mapping.json"
    mapping.write_text('{"mapping":"changed"}\n', encoding="utf-8")
    evaluator = _evaluator()
    before = evaluator.oracle.detector_queries_total
    with pytest.raises(ExternalMethodUnavailable, match="mapping SHA-256 mismatch"):
        run_external_adapter(
            evaluator,
            method_id="patchattack",
            seed=3,
            artifact_manifest_path=str(manifest),
            runner=StubExternalRunner(digest, mapping_digest),
        )
    assert evaluator.oracle.detector_queries_total == before


def test_registry_distinguishes_native_proxy_and_unavailable_close_methods():
    registry = registry_snapshot()
    assert registry["genetic_algorithm"]["fidelity"] == "native_implementation"
    assert (
        registry["fipatch_style_pso_proxy"]["fidelity"]
        == "proxy_not_fipatch_reproduction"
    )
    for method in (
        "fipatch_pso",
        "patchattack",
        "baap_2606_18318",
        "per_task_ppo",
        "wei_rl_2212_12995",
        "meta_attack_iccv21",
        "impact_de_es",
        "simulator_attack_cvpr21",
    ):
        assert registry[method]["availability"] == "requires_verified_external_runner"


def test_native_suite_checks_contract_and_emits_common_fingerprint():
    report = run_native_suite(
        StubOracle,
        budget=BudgetSpec(detector_query_limit=14, material_pixel_limit=12),
        methods=("random_search", "genetic_algorithm"),
        seed=7,
        method_configs={"genetic_algorithm": {"population_size": 3}},
    )
    assert len(report.contract_fingerprint) == 64
    assert {result.objective_id for result in report.results} == {
        "stub-objective-v1"
    }
    assert all(result.budget == report.results[0].budget for result in report.results)

    calls = 0

    def drifting_factory():
        nonlocal calls
        calls += 1
        return StubOracle(objective_id=f"drift-{calls}")

    with pytest.raises(RuntimeError, match="contract drift"):
        run_native_suite(
            drifting_factory,
            budget=BudgetSpec(14, 12),
            methods=("random_search", "forward_greedy"),
            seed=7,
        )

    with pytest.raises(ValueError, match="seed must be an integer"):
        run_native_suite(
            StubOracle,
            budget=BudgetSpec(14, 12),
            methods=("random_search",),
            seed=7.5,
        )


class SourceDetector:
    id_to_name = {1: "speed limit 25", 2: "speed limit 55"}
    target_id = 1

    def resolve_class_id(self, class_ref, *, role="class"):
        return resolve_class_id(self.id_to_name, class_ref, role=role)

    def infer_detections_batch(self, images):
        return [
            {
                "boxes": [[0.0, 0.0, float(image.width), float(image.height)]],
                "confs": [0.9],
                "clss": [1],
            }
            for image in images
        ]


def _circle_sign(size=64):
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    ImageDraw.Draw(image).ellipse(
        (2, 2, size - 3, size - 3), fill=(255, 255, 255, 255)
    )
    return image


def _traffic_env(attack_mode="disappearance"):
    sign = _circle_sign()
    attack_kwargs = {}
    if attack_mode == "untargeted_misclassification":
        attack_kwargs["allowed_alternative_classes"] = "speed limit 55"
    elif attack_mode == "targeted_misclassification":
        attack_kwargs["attack_target_class"] = "speed limit 55"
    return TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        source_class="speed limit 25",
        attack_mode=attack_mode,
        detector_instance=SourceDetector(),
        img_size=(128, 128),
        eval_K=1,
        obs_size=(64, 64),
        transform_strength=0.0,
        localization_iou_threshold=0.10,
        area_cap_frac=0.30,
        terminate_on_success=False,
        **attack_kwargs,
    )


@pytest.mark.parametrize(
    "attack_mode",
    (
        "disappearance",
        "untargeted_misclassification",
        "targeted_misclassification",
    ),
)
def test_traffic_sign_oracle_matches_env_reward_and_exact_query_area(attack_mode):
    env = _traffic_env(attack_mode)
    oracle = TrafficSignCandidateOracle(env, scene_seed=19)
    assert oracle.initial_detector_queries == 2
    assert oracle.detector_queries_per_evaluation == 2
    canonical_index = oracle.selectable_indices[0]
    row, col = divmod(canonical_index, env.Gw)
    expected_pixels = int(env._cell_pixel_areas[row, col])

    observation = oracle.evaluate((canonical_index,))
    assert oracle.detector_queries_total == 4
    assert observation.joint_success == observation.metrics["joint_success"]
    assert "objective_condition_success" in observation.metrics
    assert "objective_success" not in observation.metrics
    assert "attack_success" not in observation.metrics
    assert observation.metrics["selected_material_pixels"] == expected_pixels
    assert observation.metrics["material_fraction"] == pytest.approx(
        expected_pixels / env._sign_pixel_area
    )

    replay = _traffic_env(attack_mode)
    replay.reset(seed=19)
    valid_action = next(
        action
        for action, coord in enumerate(replay._valid_coords)
        if tuple(int(value) for value in coord) == (row, col)
    )
    _, reward, _, _, _ = replay.step(valid_action)
    assert observation.score == pytest.approx(reward)


def test_exact_material_limit_uses_integer_sign_pixels():
    assert exact_material_limit(101, 0.30) == 30
    assert exact_material_limit(100, 0.30) == 30
    with pytest.raises(ValueError):
        exact_material_limit(100, 1.01)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(tmp_path):
    with pytest.raises(StrictJSONError, match="duplicate JSON object key"):
        strict_json_loads('{"budget": 1, "budget": 2}')
    for token in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(StrictJSONError, match="non-standard"):
            strict_json_loads('{"value": ' + token + "}")
    with pytest.raises(StrictJSONError, match="strict JSON"):
        strict_json_dumps({"value": float("nan")})

    path = tmp_path / "duplicate.json"
    path.write_text('{"eval_K": 1, "eval_K": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _load_object(str(path), "environment JSON")


def test_atomic_json_publication_refuses_overwrite_and_leaves_no_temp(tmp_path):
    target = tmp_path / "report.json"
    atomic_write_json_new(target, {"run": 1})
    original = target.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_json_new(target, {"run": 2})
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".report.json.*.tmp"))

    report = run_native_suite(
        StubOracle,
        budget=BudgetSpec(10, 12),
        methods=("random_search",),
        seed=1,
        method_configs={"random_search": {"max_evaluations": 0}},
    )
    report_path = tmp_path / "comparison.json"
    report.save_json(str(report_path))
    report_bytes = report_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        report.save_json(str(report_path))
    assert report_path.read_bytes() == report_bytes


def test_environment_json_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown environment JSON keys: eval_k"):
        _validate_environment_keys({"eval_k": 3})
    _validate_environment_keys({"eval_K": 3, "action_indexing": "valid_cells"})


@pytest.mark.parametrize(
    "runner,kwargs",
    [
        (random_search, {}),
        (greedy_search, {}),
        (genetic_search, {"population_size": 2}),
        (evolution_strategy_search, {"population_size": 2}),
        (fipatch_style_pso_proxy_search, {"swarm_size": 2}),
        (cma_es_search, {"population_size": 2}),
    ],
)
def test_all_optimizers_reject_negative_max_evaluations_before_queries(
    runner, kwargs
):
    evaluator = _evaluator()
    before = evaluator.oracle.detector_queries_total
    with pytest.raises(ValueError, match="max_evaluations must be >= 0"):
        runner(evaluator, seed=1, max_evaluations=-1, **kwargs)
    assert evaluator.oracle.detector_queries_total == before


def test_optimizer_rejects_nonfinite_configuration_before_queries():
    evaluator = _evaluator()
    before = evaluator.oracle.detector_queries_total
    with pytest.raises(ValueError, match="sigma must be a finite number"):
        evolution_strategy_search(
            evaluator, seed=1, population_size=2, sigma=float("nan")
        )
    assert evaluator.oracle.detector_queries_total == before


def test_candidates_require_exact_integer_indices():
    evaluator = _evaluator()
    with pytest.raises(ValueError, match="selected index must be an integer"):
        evaluator.evaluate((1.0,))
    with pytest.raises(ValueError, match="not a boolean"):
        evaluator.evaluate((True,))


def test_nonfinite_oracle_metrics_abort_instead_of_becoming_json_null():
    class NonfiniteMetricOracle(StubOracle):
        def evaluate(self, selected_indices):
            self.detector_queries_total += self.detector_queries_per_evaluation
            return OracleObservation(
                score=0.0,
                joint_success=False,
                metrics={"margin": float("inf")},
            )

    evaluator = BudgetedEvaluator(
        NonfiniteMetricOracle(), BudgetSpec(10, 12)
    )
    with pytest.raises(ValueError, match="non-finite numeric"):
        evaluator.evaluate(())
    assert evaluator.queries_used == StubOracle.initial_detector_queries + 2


def test_material_budget_must_match_environment_objective_exactly():
    oracle = TrafficSignCandidateOracle(_traffic_env(), scene_seed=19)
    assert oracle.objective_material_pixel_limit is not None
    mismatched = oracle.objective_material_pixel_limit - 1
    try:
        with pytest.raises(ValueError, match="does not match"):
            BudgetedEvaluator(
                oracle,
                BudgetSpec(
                    detector_query_limit=10,
                    material_pixel_limit=mismatched,
                ),
            )
    finally:
        oracle.close()


def test_suite_closes_every_oracle_on_success_and_contract_failure():
    created = []

    def factory():
        oracle = StubOracle()
        created.append(oracle)
        return oracle

    run_native_suite(
        factory,
        budget=BudgetSpec(10, 12),
        methods=("random_search", "forward_greedy"),
        seed=1,
        method_configs={
            "random_search": {"max_evaluations": 1},
            "forward_greedy": {"max_evaluations": 1},
        },
    )
    assert len(created) == 2
    assert all(oracle.closed for oracle in created)

    drifted = []

    def drift_factory():
        oracle = StubOracle(f"objective-{len(drifted)}")
        drifted.append(oracle)
        return oracle

    with pytest.raises(RuntimeError, match="contract drift"):
        run_native_suite(
            drift_factory,
            budget=BudgetSpec(10, 12),
            methods=("random_search", "forward_greedy"),
            seed=1,
            method_configs={
                "random_search": {"max_evaluations": 0},
                "forward_greedy": {"max_evaluations": 0},
            },
        )
    assert len(drifted) == 2
    assert all(oracle.closed for oracle in drifted)


def test_suite_rejects_config_typos_before_constructing_oracle():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return StubOracle()

    with pytest.raises(ValueError, match="unknown config keys"):
        run_native_suite(
            factory,
            budget=BudgetSpec(10, 12),
            methods=("random_search",),
            seed=1,
            method_configs={"random_search": {"max_evaluation": 2}},
        )
    assert calls == 0


def test_traffic_oracle_close_is_idempotent_and_blocks_evaluation():
    env = _traffic_env()
    detector = env.det
    detector.closed = False
    detector.close = lambda: setattr(detector, "closed", True)
    oracle = TrafficSignCandidateOracle(env, scene_seed=19)
    oracle.close()
    oracle.close()
    assert detector.closed
    with pytest.raises(RuntimeError, match="oracle is closed"):
        oracle.evaluate(())


def test_traffic_oracle_constructor_failure_closes_owned_environment():
    class Detector:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class InvalidAdaptiveEnv:
        eval_K_min = 1
        eval_K_max = 2

        def __init__(self):
            self.det = Detector()
            self.closed = False

        def close(self):
            self.closed = True

    env = InvalidAdaptiveEnv()
    with pytest.raises(ValueError, match="fixed EOT"):
        TrafficSignCandidateOracle(env, scene_seed=1)
    assert env.closed
    assert env.det.closed


def test_cli_artifact_identity_hashes_weights_and_entire_background_set(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"checkpoint-v1")
    backgrounds = tmp_path / "backgrounds"
    backgrounds.mkdir()
    Image.new("RGB", (8, 8), "red").save(backgrounds / "a.png")
    Image.new("RGB", (8, 8), "blue").save(backgrounds / "b.png")
    config = _default_cfg_for_env(
        {
            "detector": "yolo",
            "yolo_weights": str(weights),
            "bg_mode": "dataset",
            "bgdir": str(backgrounds),
        }
    )
    identity = _experiment_artifact_identity(config)
    assert identity["detector"]["weights"]["sha256"] == hashlib.sha256(
        b"checkpoint-v1"
    ).hexdigest()
    assert [row["name"] for row in identity["background_collection"]["files"]] == [
        "a.png",
        "b.png",
    ]
    first_hashes = [
        row["sha256"] for row in identity["background_collection"]["files"]
    ]
    Image.new("RGB", (8, 8), "green").save(backgrounds / "b.png")
    changed = _experiment_artifact_identity(config)
    assert [
        row["sha256"] for row in changed["background_collection"]["files"]
    ] != first_hashes
