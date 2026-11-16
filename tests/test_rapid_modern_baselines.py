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
from baselines.modern.attacks import fipatch, impact
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
    from types import SimpleNamespace
    args = SimpleNamespace(output=str(output), compare_root=str(tmp_path/"compare"),
                           detector_names=["yolo8"], seed_values=[1000], method_names=["FIPatch", "IMPACT"])
    runner.export_report(args)
    report = json.loads((output/"comparison.json").read_text())
    random = next(r for r in report if r["method"]=="Random")
    assert random["record_count"] == 50 and random["mean_minutes"] == 1
    partial_row = next(r for r in report if r["method"]=="FIPatch")
    assert partial_row["completed_seeds"] == 0 and partial_row["ASR"] == ""


class SyntheticOracle:
    """Exercises real pinned optimizer kernels, never produces detector results."""
    def __init__(self, budget=64):
        self.shape = (32, 32)
        self.grid_cell = 4
        self.sign_pixels = 32*32
        self.area_target = .25
        self.candidate_limit = budget
        self.evaluations = 0
        self.scores = []

    def evaluate(self, mask, rgb):
        if self.evaluations >= self.candidate_limit:
            raise SearchStopped("query_limit")
        assert mask.shape == self.shape and rgb.shape == (*self.shape, 3)
        assert np.isfinite(rgb).all()
        self.evaluations += 1
        score = float(mask.mean() - np.mean(rgb)/255)
        self.scores.append(score)
        return {"score": score}


@pytest.mark.parametrize("method", ["FIPatch", "IMPACT"])
def test_pinned_kernels_respect_external_candidate_cap(method):
    directory = ROOT/"_runs/rapid_baseline_test_upstream"
    if not directory.exists():
        pytest.skip("Run prepare-only once to populate hash-pinned author sources")
    kernels = load_kernels(directory)
    oracle = SyntheticOracle()
    with pytest.raises(SearchStopped, match="query_limit"):
        if method == "FIPatch":
            fipatch(oracle, kernels[0], seed=1000)
        else:
            impact(oracle, kernels[1], seed=1000, population=8)
    assert oracle.evaluations == 64


def test_tampered_upstream_source_is_rejected(tmp_path):
    (tmp_path/"fipatch_attack.py").write_text("print('not the official source')")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        fetch_sources(tmp_path)
