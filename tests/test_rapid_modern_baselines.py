"""Budget/provenance tests for the rapid two-method runner; no paper results."""
from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.modern.oracle import evaluate_metrics, RecordingDetector, SearchStopped
from baselines.modern.attacks import fipatch, impact, circle_patch, repair_tiles, tile_support
from baselines.modern.upstream import fetch_sources, load_kernels


def runner_module():
    spec = importlib.util.spec_from_file_location("rapid_runner", ROOT/"tools/run_rapid_modern_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_signed_daylight_and_soft_area_budget_are_separate():
    row = evaluate_metrics(.95, .96, .96, .15, .4, {"area_cap_mode": "soft", "area_cap_frac": .3})
    assert row["drop_day"] == pytest.approx(-.01)
    assert row["success"] and row["area_cap_exceeded"]
    assert not row["joint_success"]
    assert row["score"] < 0


def test_clean_failure_is_not_attack_success():
    row = evaluate_metrics(.1, .1, .1, .0, .1, {})
    assert not row["clean_eligible"] and not row["success"]


def test_query_counter_counts_images_not_batches_and_does_not_grow_memory():
    class Detector:
        def infer_detections_batch(self, images):
            return [{"target_conf": .5} for _ in images]
    recorder = RecordingDetector(Detector())
    for _ in range(20):
        recorder.infer_detections_batch([None]*6)
    assert recorder.images == 120
    assert len(recorder.records) == 64


def test_saved_methods_are_never_added_to_executable_methods(monkeypatch):
    runner = runner_module()
    monkeypatch.setattr(sys, "argv", ["runner", "--methods", "DIRP"])
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_deadline_cannot_exceed_fourteen_hours(monkeypatch):
    runner = runner_module()
    monkeypatch.setattr(sys, "argv", ["runner", "--hours", "14"])
    with pytest.raises(SystemExit):
        runner.parse_args()


def test_first_run_does_not_reject_its_own_cache_directory(tmp_path, monkeypatch):
    from baselines.modern import upstream
    runner = runner_module()
    output = tmp_path/"new_run"
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["runner", "--prepare-only", "--output", str(output)])
    monkeypatch.setattr(runner, "preflight", lambda args: {})
    monkeypatch.setattr(upstream, "fetch_sources", lambda directory: {})
    assert runner.main() == 0
    assert (output/"manifest.json").exists()


def test_parent_watchdog_terminates_a_stalled_child():
    runner = runner_module()
    started = time.monotonic()
    code = runner.supervised([sys.executable, "-c", "import time; time.sleep(120)"], started+0.15)
    assert code == 124
    assert time.monotonic()-started < 10


def test_runtime_random_is_per_recorded_trial_and_partial_is_missing(tmp_path):
    runner = runner_module()
    compare = tmp_path/"compare/yolo8_N5_seed1000"
    random_path = compare/"random/run1"
    random_path.mkdir(parents=True)
    records = [{"success": True, "after_conf": .1, "area_frac": .2, "runtime_sec": 60}]*50
    (random_path/"summary.json").write_text(json.dumps({"trials_detail": records}))
    saved = {"episodes": {
        "ppo": [{"seed": 1000, "success": True, "after_conf": .1, "area_frac": .2, "runtime_sec": 5}],
        "greedy": [{"seed": 1000, "success": True, "after_conf": .1, "area_frac": .2, "runtime_sec": 500}],
        "random": [{"seed": 1000, "run_dir": "/old/random/run1"}]}}
    (compare/"compare_summary.json").write_text(json.dumps(saved))
    output = tmp_path/"output"
    partial = output/"yolo8/1000/FIPatch"
    partial.mkdir(parents=True)
    (partial/"result.json").write_text(json.dumps({"status": "wall_time_limit", "best": {"success": True, "after_conf": .1, "area_frac": .2}, "runtime_sec": 1}))
    completed = output/"yolo8/1000/IMPACT"
    completed.mkdir(parents=True)
    (completed/"result.json").write_text(json.dumps({"status": "query_limit",
        "best": {"success": True, "joint_success": False, "after_conf": .1,
                 "area_frac": .5, "drop_on": .8, "clean_eligible": True},
        "runtime_sec": 2, "minimum_evaluated_confidence": .02, "zero_area_candidates": 0}))
    from types import SimpleNamespace
    args = SimpleNamespace(output=str(output), compare_root=str(tmp_path/"compare"),
                           detector_names=["yolo8"], seed_values=[1000], method_names=["FIPatch", "IMPACT"])
    runner.export_report(args)
    report = json.loads((output/"comparison.json").read_text())
    random = next(r for r in report if r["method"]=="Random")
    assert random["record_count"] == 50 and random["mean_minutes"] == 1
    partial_row = next(r for r in report if r["method"]=="FIPatch")
    assert partial_row["completed_seeds"] == 0 and partial_row["ASR"] == ""
    complete_row = next(r for r in report if r["method"]=="IMPACT")
    assert complete_row["ASR"] == 1 and complete_row["joint_ASR"] == 0
    assert complete_row["mean_minimum_evaluated_confidence"] == .02
    assert random["joint_ASR"] == ""


class SyntheticOracle:
    """Exercises real pinned optimizer kernels, never produces detector results."""
    def __init__(self, budget=64, color_mode="rgb"):
        self.shape = (32, 32)
        self.grid_cell = 4
        self.sign_pixels = 32*32
        self.sign_mask = np.ones(self.shape, dtype=bool)
        self.config = {"color_mode": color_mode}
        self.material_rgb = np.asarray([230, 255, 0], dtype=np.float32)
        self.area_target = .25
        self.candidate_limit = budget
        self.evaluations = 0
        self.scores = []
        self.stages = set()

    def evaluate(self, mask, rgb):
        if self.evaluations >= self.candidate_limit:
            raise SearchStopped("query_limit")
        assert mask.shape == self.shape and rgb.shape == (*self.shape, 3)
        assert np.isfinite(rgb).all()
        assert (mask & self.sign_mask).any()
        if self.config["color_mode"] == "material":
            assert (rgb[mask & self.sign_mask] == self.material_rgb).all()
        self.stages.add(self.search_stage)
        self.evaluations += 1
        score = float(mask.mean() - np.mean(rgb)/255)
        self.scores.append(score)
        return {"score": score}


@pytest.mark.parametrize("method", ["FIPatch", "IMPACT"])
@pytest.mark.parametrize("color_mode", ["rgb", "material"])
def test_pinned_kernels_respect_external_candidate_cap(method, color_mode):
    directory = ROOT/"_runs/rapid_baseline_test_upstream"
    if not directory.exists():
        pytest.skip("Run prepare-only once to populate hash-pinned author sources")
    kernels = load_kernels(directory)
    oracle = SyntheticOracle(color_mode=color_mode)
    with pytest.raises(SearchStopped, match="query_limit"):
        if method == "FIPatch":
            fipatch(oracle, kernels[0], seed=1000)
        else:
            impact(oracle, kernels[1], seed=1000, population=8)
    assert oracle.evaluations == 64
    if color_mode == "rgb":
        assert len(oracle.stages) == 2


def test_tampered_upstream_source_is_rejected(tmp_path):
    (tmp_path/"fipatch_attack.py").write_text("print('not the official source')")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        fetch_sources(tmp_path)


def test_circle_in_transparent_corner_is_repaired_to_real_sign():
    support = np.zeros((64, 64), dtype=bool)
    support[20:44, 20:44] = True
    for center in [(0, 0), (63, 63), (0, 63), (63, 0)]:
        mask, rgb = circle_patch(support.shape, support, [center], 5, [[255, 80, 0]])
        assert mask.any() and not (mask & ~support).any()
        assert (rgb[mask] == [255, 80, 0]).all()


def test_suppression_diagnostic_changes_search_not_success_or_area_metrics():
    original = evaluate_metrics(.95, .96, .94, .5, .2, {})
    pilot = evaluate_metrics(.95, .96, .94, .5, .2, {"search_objective": "suppression"})
    assert pilot["optimization_score"] == -.5
    assert original["optimization_score"] == original["score"]
    for key in original.keys() - {"optimization_score"}:
        assert original[key] == pilot[key]


def test_off_sign_tile_repair_preserves_colors_and_cardinality():
    code = np.zeros(16, dtype=int)
    code[[0, 5, 15]] = 1
    colors = np.asarray([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
    repaired, content = repair_tiles(code, colors, np.asarray([5, 6, 9, 10]), 4)
    assert np.flatnonzero(repaired).tolist() == [5, 6, 10]
    assert content.tolist() == [colors[1].tolist(), colors[0].tolist(), colors[2].tolist()]


def test_tile_support_omits_transparent_cells():
    support = np.zeros((32, 32), dtype=bool)
    support[8:24, 8:24] = True
    side, valid, areas = tile_support(support, 4)
    assert side == 8 and len(valid) == 16
    assert (areas[valid] == 16).all()


@pytest.mark.parametrize("threshold", [.3, .6, .9])
def test_impact_eligible_cells_match_actual_dirp_grid_including_edges(threshold):
    from PIL import Image
    from envs.stop_sign_grid_env import TrafficSignGridEnv
    # Run the real grid builder without constructing or querying a detector.
    env = object.__new__(TrafficSignGridEnv)
    env.sign_rgba_day = Image.open(ROOT/"data/stop_sign.png").convert("RGBA")
    env._sign_alpha = env.sign_rgba_day.getchannel("A")
    env.grid_cell_px, env.cell_cover_thresh = 16, threshold
    env._build_grid_index()
    mask = np.asarray(env._sign_alpha) > 0
    side, valid, areas = tile_support(mask, 16, threshold)
    assert side == env.Gw == env.Gh
    assert np.array_equal(valid, np.flatnonzero(env._valid_cells))
    assert np.array_equal(areas, env._cell_pixel_areas.reshape(-1))
    if threshold == .6:
        assert len(valid) == 221
        assert set([101, 118, 135, 152, 277, 278, 279, 280, 281, 282]).issubset(valid)


def test_partial_cells_use_their_cropped_size_not_full_tile_size():
    # A 5x5 opaque sign has four valid 4px cells, including 1x4 and 1x1 edges.
    side, valid, areas = tile_support(np.ones((5, 5), dtype=bool), 4)
    assert side == 2 and valid.tolist() == [0, 1, 2, 3]
    assert areas.tolist() == [16, 4, 4, 1]


def test_fixed_query_policy_and_material_mode_reach_workers(monkeypatch):
    runner = runner_module()
    monkeypatch.setattr(sys, "argv", ["runner", "--budget-policy", "fixed",
        "--color-mode", "material", "--search-areas", "0.25,0.5", "--queries", "9006"])
    args = runner.parse_args()
    command = runner.child_command(args, "attack", 123.0, "yolo8")
    assert args.search_area_levels == [.25, .5]
    assert command[command.index("--queries")+1] == "9006"
    assert command[command.index("--budget-policy")+1] == "fixed"
    assert command[command.index("--color-mode")+1] == "material"
