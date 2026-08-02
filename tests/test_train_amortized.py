from types import SimpleNamespace

import pytest

from train_amortized import (
    _transport_paint,
    _validate_calibration,
    _validate_research_splits,
)
from utils.task_manifest import load_task_manifest


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
