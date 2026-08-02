import pytest

from detectors.class_names import normalize_class_name, resolve_class_id


def test_class_resolution_is_punctuation_insensitive_but_exact():
    names = {3: "Speed Limit 25", 8: "STOP-sign"}
    assert normalize_class_name("speed_limit-25") == "speedlimit25"
    assert resolve_class_id(names, "speed_limit-25") == 3
    assert resolve_class_id(names, "stop sign") == 8
    assert resolve_class_id(names, "3") == 3
    assert resolve_class_id({}, "17") == 17


def test_numeric_id_must_exist_when_detector_exposes_a_label_map():
    with pytest.raises(ValueError, match="absent from the detector label map"):
        resolve_class_id({3: "Speed Limit 25"}, "17", role="source class")


def test_missing_label_fails_instead_of_falling_back_to_coco_id():
    with pytest.raises(ValueError, match="Could not resolve source class"):
        resolve_class_id({11: "stop sign"}, "speed limit 25", role="source class")
