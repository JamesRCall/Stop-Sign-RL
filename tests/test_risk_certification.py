import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import certify_attack_results as certification_cli
from utils.risk_certification import (
    CLAIM_NAMES,
    calibrate_prefixes,
    canonical_sha256,
    certify_selected_prefix,
    parse_protocol,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_cli_json_reader_rejects_duplicate_keys_and_nonfinite_constants(tmp_path):
    duplicate = Path(tmp_path) / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        certification_cli._read_json(duplicate, label="plan")

    nonfinite = Path(tmp_path) / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        certification_cli._read_json(nonfinite, label="rows")


def _fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator}


def _trial_rows(phase: str, task_id: str, count: int) -> list[dict]:
    return [
        {
            "trial_id": f"{phase}_{task_id}_{index}",
            "sample_sha256": _digest(f"{phase}:{task_id}:{index}"),
        }
        for index in range(count)
    ]


def _protocol(*, task_count: int = 1, prefix_count: int = 2, trials: int = 4) -> dict:
    tasks = []
    for task_index in range(task_count):
        task_id = f"task_{task_index}"
        tasks.append(
            {
                "task_id": task_id,
                "detector_sha256": _digest(f"detector:{task_id}"),
                "calibration_trials": _trial_rows(
                    "calibration", task_id, trials
                ),
                "certification_trials": _trial_rows(
                    "certification", task_id, trials
                ),
            }
        )
    prefixes = []
    for prefix_index in range(prefix_count):
        prefix_id = f"prefix_{prefix_index + 1}"
        prefixes.append(
            {
                "prefix_id": prefix_id,
                "order": prefix_index + 1,
                "task_patterns": [
                    {
                        "task_id": task["task_id"],
                        "pattern_sha256": _digest(
                            f"pattern:{prefix_id}:{task['task_id']}"
                        ),
                        "area": {
                            "selected_pixels": prefix_index + 1,
                            "sign_pixels": 10,
                        },
                    }
                    for task in tasks
                ],
            }
        )
    calibration_row_count = prefix_count * task_count * trials
    certification_row_count = task_count * trials
    return {
        "schema_version": 1,
        "protocol_id": "unit_test_protocol",
        "prefix_family_sha256": _digest("unit-test-prefix-family"),
        "estimand": {
            "population_description": "preregistered synthetic traffic-sign scenes",
            "sampling_unit": "independent_scene_transform_pair",
            "task_aggregation": "all_tasks",
            "fixed_sample_size": True,
        },
        "familywise_alpha": {
            "calibration": _fraction(1, 20),
            "certification": _fraction(1, 20),
        },
        "claims": {name: _fraction(1, 5) for name in CLAIM_NAMES},
        "area_cap": _fraction(1, 2),
        "tasks": tasks,
        "prefixes": prefixes,
        "excluded_sample_sha256": {
            "training": [_digest("train:0")],
            "development": [_digest("development:0")],
        },
        "query_accounting": {
            "offline_training_detector_image_queries": 100,
            "development_selection_detector_image_queries": 20,
            "prefix_generation_detector_image_queries": 0,
            "expected_calibration_detector_image_queries": calibration_row_count * 4,
            "expected_certification_detector_image_queries": certification_row_count * 4,
        },
    }


def _outcomes(*, passing: bool) -> dict:
    if passing:
        return {name: True for name in CLAIM_NAMES}
    return {
        "joint_success": False,
        "attack_success": False,
        "day_preservation": True,
        "clean_eligibility": True,
    }


def _phase_results(
    protocol: dict,
    *,
    phase: str,
    passing_prefixes: set[str],
    selection_sha256: str | None = None,
    task_failures: set[tuple[str, str]] | None = None,
) -> dict:
    task_failures = task_failures or set()
    prefixes = (
        protocol["prefixes"]
        if phase == "calibration"
        else [
            prefix
            for prefix in protocol["prefixes"]
            if prefix["prefix_id"] in passing_prefixes
        ]
    )
    rows = []
    for prefix in prefixes:
        pattern_map = {
            item["task_id"]: item for item in prefix["task_patterns"]
        }
        for task in protocol["tasks"]:
            trial_key = f"{phase}_trials"
            should_pass = (
                prefix["prefix_id"] in passing_prefixes
                and (prefix["prefix_id"], task["task_id"]) not in task_failures
            )
            for trial in task[trial_key]:
                pattern = pattern_map[task["task_id"]]
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "prefix_id": prefix["prefix_id"],
                        "trial_id": trial["trial_id"],
                        "sample_sha256": trial["sample_sha256"],
                        "pattern_sha256": pattern["pattern_sha256"],
                        "detector_sha256": task["detector_sha256"],
                        "outcomes": _outcomes(passing=should_pass),
                        "area": copy.deepcopy(pattern["area"]),
                        "detector_image_queries": 4,
                    }
                )
    payload = {
        "schema_version": 1,
        "protocol_sha256": canonical_sha256(protocol),
        "phase": phase,
        "rows": rows,
    }
    if phase == "certification":
        payload["selection_sha256"] = selection_sha256
    return payload


def test_two_phase_selection_and_final_certification_pass():
    protocol = _protocol()
    calibration_rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    calibration = calibrate_prefixes(protocol, calibration_rows)

    assert calibration["selection"]["selected_prefix_id"] == "prefix_2"
    assert calibration["simultaneous_inference"]["comparison_count"] == 8
    assert calibration["candidate_family"]["all_candidates_evaluated"] is True
    assert calibration["candidate_family"]["success_monotonicity_assumed"] is False
    assert calibration["final_certificate_issued"] is False

    certification_rows = _phase_results(
        protocol,
        phase="certification",
        passing_prefixes={"prefix_2"},
        selection_sha256=calibration["selection_sha256"],
    )
    certificate = certify_selected_prefix(protocol, calibration, certification_rows)

    assert certificate["certificate_status"] == "pass"
    assert certificate["certificate_pass"] is True
    assert certificate["simultaneous_inference"]["comparison_count"] == 4
    assert certificate["claims_formal_or_all_world_guarantee"] is False
    assert certificate["queries"]["full_lifecycle_detector_image_queries"] == 168


def test_smallest_passing_prefix_is_selected_without_short_circuiting():
    protocol = _protocol(prefix_count=3)
    calibration_rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2", "prefix_3"},
    )
    report = calibrate_prefixes(protocol, calibration_rows)
    assert report["selection"]["selected_prefix_id"] == "prefix_2"
    assert len(report["prefixes"]) == 3
    assert [item["prefix_id"] for item in report["prefixes"]] == [
        "prefix_1",
        "prefix_2",
        "prefix_3",
    ]


def test_nonmonotone_candidate_success_is_measured_not_inferred():
    protocol = _protocol(prefix_count=3)
    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_1", "prefix_3"},
    )
    report = calibrate_prefixes(protocol, rows)
    observed = {
        item["prefix_id"]: item["all_tasks_pass"] for item in report["prefixes"]
    }
    assert observed == {
        "prefix_1": True,
        "prefix_2": False,
        "prefix_3": True,
    }
    assert report["selection"]["selected_prefix_id"] == "prefix_1"


def test_one_bad_task_defeats_pooled_success_and_sets_worst_task():
    protocol = _protocol(task_count=2, trials=4)
    calibration_rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
        task_failures={("prefix_2", "task_1")},
    )
    report = calibrate_prefixes(protocol, calibration_rows)
    prefix_two = next(item for item in report["prefixes"] if item["prefix_id"] == "prefix_2")
    assert prefix_two["all_tasks_pass"] is False
    assert prefix_two["worst_task_lower_bounds"]["joint_success"]["task_ids"] == [
        "task_1"
    ]
    assert report["selection"]["status"] == "no_prefix_passed"


def test_calibration_and_certification_results_must_be_disjoint():
    protocol = _protocol()
    protocol["tasks"][0]["certification_trials"][0]["sample_sha256"] = protocol[
        "tasks"
    ][0]["calibration_trials"][0]["sample_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        parse_protocol(protocol)


def test_unknown_and_missing_fields_fail_closed():
    protocol = _protocol()
    protocol["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown=unexpected"):
        parse_protocol(protocol)

    protocol = _protocol()
    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    del rows["rows"][0]["outcomes"]["day_preservation"]
    with pytest.raises(ValueError, match="missing=day_preservation"):
        calibrate_prefixes(protocol, rows)


def test_duplicate_tasks_trials_and_hashes_are_rejected():
    protocol = _protocol()
    protocol["tasks"].append(copy.deepcopy(protocol["tasks"][0]))
    with pytest.raises(ValueError, match="duplicate task_id"):
        parse_protocol(protocol)

    protocol = _protocol()
    protocol["tasks"][0]["calibration_trials"].append(
        copy.deepcopy(protocol["tasks"][0]["calibration_trials"][0])
    )
    with pytest.raises(ValueError, match="duplicate trial_id"):
        parse_protocol(protocol)


def test_exact_area_constraint_and_joint_truth_table_are_enforced():
    protocol = _protocol()
    # Prefix 2 has area 2/10.  A 1/10 cap makes it an exact material failure.
    protocol["area_cap"] = _fraction(1, 10)
    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    with pytest.raises(ValueError, match="joint_success must equal"):
        calibrate_prefixes(protocol, rows)

    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes=set(),
    )
    report = calibrate_prefixes(protocol, rows)
    prefix_two = next(item for item in report["prefixes"] if item["prefix_id"] == "prefix_2")
    assert prefix_two["tasks"][0]["exact_area"]["passes"] is False
    assert prefix_two["tasks"][0]["exact_area"]["statistical_alpha_consumed"] is False


def test_query_total_and_protocol_binding_must_match_exactly():
    protocol = _protocol()
    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    rows["rows"][0]["detector_image_queries"] -= 1
    with pytest.raises(ValueError, match="query total mismatch"):
        calibrate_prefixes(protocol, rows)

    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    rows["protocol_sha256"] = _digest("wrong protocol")
    with pytest.raises(ValueError, match="does not match"):
        calibrate_prefixes(protocol, rows)


def test_certification_cannot_use_unselected_prefix_or_calibration_rows():
    protocol = _protocol()
    calibration_rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    selection = calibrate_prefixes(protocol, calibration_rows)
    wrong_rows = _phase_results(
        protocol,
        phase="certification",
        passing_prefixes={"prefix_1"},
        selection_sha256=selection["selection_sha256"],
    )
    with pytest.raises(ValueError, match="exactly match"):
        certify_selected_prefix(protocol, selection, wrong_rows)

    calibration_rows["selection_sha256"] = selection["selection_sha256"]
    with pytest.raises(ValueError, match="phase"):
        certify_selected_prefix(protocol, selection, calibration_rows)


def test_selection_report_integrity_is_verified():
    protocol = _protocol()
    rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    selection = calibrate_prefixes(protocol, rows)
    selection["selection"]["selected_prefix_id"] = "prefix_1"
    certification_rows = _phase_results(
        protocol,
        phase="certification",
        passing_prefixes={"prefix_2"},
        selection_sha256=selection["selection_sha256"],
    )
    with pytest.raises(ValueError, match="integrity hash"):
        certify_selected_prefix(protocol, selection, certification_rows)


def test_canonical_protocol_hash_ignores_object_key_order():
    protocol = _protocol()
    reversed_root = dict(reversed(list(protocol.items())))
    assert canonical_sha256(protocol) == canonical_sha256(reversed_root)
    assert parse_protocol(protocol).canonical_sha256 == canonical_sha256(protocol)


def test_cli_hash_calibrate_certify_and_no_overwrite(tmp_path, capsys):
    protocol = _protocol()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(protocol), encoding="utf-8")
    assert certification_cli.main(["hash-plan", "--plan", str(plan_path)]) == 0
    assert canonical_sha256(protocol) in capsys.readouterr().out

    calibration_rows = _phase_results(
        protocol,
        phase="calibration",
        passing_prefixes={"prefix_2"},
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_rows), encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    assert (
        certification_cli.main(
            [
                "calibrate",
                "--plan",
                str(plan_path),
                "--rows",
                str(calibration_path),
                "--out",
                str(selection_path),
            ]
        )
        == 0
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    certification_rows = _phase_results(
        protocol,
        phase="certification",
        passing_prefixes={"prefix_2"},
        selection_sha256=selection["selection_sha256"],
    )
    certification_path = tmp_path / "certification.json"
    certification_path.write_text(json.dumps(certification_rows), encoding="utf-8")
    certificate_path = tmp_path / "certificate.json"
    assert (
        certification_cli.main(
            [
                "certify",
                "--plan",
                str(plan_path),
                "--selection",
                str(selection_path),
                "--rows",
                str(certification_path),
                "--out",
                str(certificate_path),
            ]
        )
        == 0
    )
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    assert certificate["certificate_status"] == "pass"
    assert certificate["input_files"]["protocol_file_sha256"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        certification_cli.main(
            [
                "certify",
                "--plan",
                str(plan_path),
                "--selection",
                str(selection_path),
                "--rows",
                str(certification_path),
                "--out",
                str(certificate_path),
            ]
        )
