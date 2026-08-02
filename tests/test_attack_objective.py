from dataclasses import replace

import pytest

from envs.attack_objective import (
    AttackObjectiveConfig,
    aggregate_metrics,
    attack_success,
    summarize_detection,
)


SIGN_BOX = [0, 0, 100, 100]


def _det(boxes, confs, classes):
    return {"boxes": boxes, "confs": confs, "clss": classes}


def _config(mode="disappearance", **kwargs):
    return AttackObjectiveConfig(
        mode=mode,
        source_conf_threshold=0.20,
        target_conf_threshold=0.40,
        min_success_rate=0.75,
        localization_iou_threshold=0.30,
        require_source_suppression=True,
        **kwargs,
    )


def test_unrelated_background_detection_is_not_misclassification():
    row = summarize_detection(
        _det([[200, 200, 260, 260]], [0.99], [7]),
        SIGN_BOX,
        source_class_id=1,
        attack_target_id=2,
        config=_config("untargeted_misclassification"),
    )
    assert row.top_class is None
    assert row.disappearance_success
    assert not row.untargeted_success
    assert not row.targeted_success


def test_localized_wrong_class_requires_source_suppression():
    config = _config("untargeted_misclassification")
    suppressed = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX], [0.10, 0.80], [1, 7]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    retained = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX], [0.30, 0.80], [1, 7]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    assert suppressed.untargeted_success
    assert not retained.untargeted_success


def test_untargeted_allowlist_rejects_unregistered_wrong_labels():
    config = _config(
        "untargeted_misclassification",
        allowed_alternative_class_ids=(2, 3),
    )
    unregistered = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX], [0.10, 0.95], [1, 7]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    registered = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX], [0.10, 0.80], [1, 2]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    assert unregistered.alternative_class is None
    assert not unregistered.untargeted_success
    assert registered.alternative_class == 2
    assert registered.untargeted_success


def test_targeted_class_must_be_localized_and_top_ranked():
    config = _config("targeted_misclassification")
    wrong_top = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX, SIGN_BOX], [0.10, 0.60, 0.70], [1, 2, 7]),
        SIGN_BOX,
        1,
        2,
        config,
    )
    target_top = summarize_detection(
        _det([SIGN_BOX, SIGN_BOX, SIGN_BOX], [0.10, 0.80, 0.70], [1, 2, 7]),
        SIGN_BOX,
        1,
        2,
        config,
    )
    target_elsewhere = summarize_detection(
        _det([SIGN_BOX, [200, 200, 260, 260]], [0.10, 0.99], [1, 2]),
        SIGN_BOX,
        1,
        2,
        config,
    )
    assert not wrong_top.targeted_success
    assert target_top.targeted_success
    assert not target_elsewhere.targeted_success


def test_eot_success_boundary_is_inclusive():
    config = _config("targeted_misclassification")
    success_row = summarize_detection(
        _det([SIGN_BOX], [0.80], [2]), SIGN_BOX, 1, 2, config
    )
    failure_row = summarize_detection(
        _det([SIGN_BOX], [0.80], [7]), SIGN_BOX, 1, 2, config
    )
    metrics = aggregate_metrics([success_row, success_row, success_row, failure_row])
    assert metrics.targeted_rate == pytest.approx(0.75)
    assert attack_success(metrics, config)
    assert not attack_success(metrics, replace(config, min_success_rate=0.76))


def test_empty_and_malformed_detections_do_not_crash_or_fake_misclassification():
    config = _config("untargeted_misclassification")
    empty = summarize_detection({}, SIGN_BOX, 1, None, config)
    malformed = summarize_detection(
        _det([SIGN_BOX, [1, 2]], [0.9], [7, 8]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    assert empty.disappearance_success
    assert not empty.untargeted_success
    assert malformed.top_class == 7
    assert malformed.untargeted_success


def test_source_box_below_localization_threshold_counts_as_drift_not_wrong_class():
    config = _config("untargeted_misclassification")
    row = summarize_detection(
        _det([[80, 80, 180, 180]], [0.95], [1]),
        SIGN_BOX,
        1,
        None,
        config,
    )
    assert row.source_conf == 0.0
    assert row.source_iou > 0.0
    assert row.disappearance_success
    assert not row.untargeted_success


def test_targeted_mode_requires_target_id_at_environment_boundary():
    with pytest.raises(ValueError):
        AttackObjectiveConfig(mode="not_a_mode")
