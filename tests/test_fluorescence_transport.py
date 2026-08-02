import copy
import json
from pathlib import Path

import numpy as np
import pytest

from utils.fluorescence_transport import (
    CalibrationError,
    ExcitationEmissionModel,
    MeasuredEmissionResponseModel,
    compute_fluorescence_transport,
    fluorescence_transport_schema_path,
    load_fluorescence_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_FIXTURE = (
    ROOT / "data" / "synthetic" / "fluorescence_transport_v1.synthetic.json"
)


def _fixture_document():
    return json.loads(SYNTHETIC_FIXTURE.read_text(encoding="utf-8"))


def _write_document(tmp_path, document, name="calibration.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_versioned_schema_and_fixture_are_explicitly_synthetic():
    schema = json.loads(
        fluorescence_transport_schema_path().read_text(encoding="utf-8")
    )
    calibration = load_fluorescence_calibration(SYNTHETIC_FIXTURE)

    assert schema["$id"] == "urn:stop-sign-model:schema:fluorescence-transport:1.0.0"
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"
    assert calibration.schema_version == "1.0.0"
    assert calibration.calibration_type == "synthetic_non_empirical"
    assert not calibration.provenance.is_empirical
    assert calibration.ambient_illuminant_name.startswith("synthetic")
    assert calibration.uv_trigger_name.startswith("synthetic")
    assert "SYNTHETIC NON-EMPIRICAL DATA" in calibration.provenance.non_empirical_notice
    assert isinstance(calibration.material.fluorescence_model, ExcitationEmissionModel)
    assert len(calibration.canonical_sha256) == 64
    assert len(calibration.source_sha256) == 64


def test_transport_is_seed_deterministic_and_has_regression_values():
    calibration = load_fluorescence_calibration(SYNTHETIC_FIXTURE)

    first = compute_fluorescence_transport(calibration, seed=42)
    repeated = compute_fluorescence_transport(calibration, seed=42)
    other_seed = compute_fluorescence_transport(calibration, seed=7)

    assert first.to_dict() == repeated.to_dict()
    assert first.uncertainty_draw.to_dict() != other_seed.uncertainty_draw.to_dict()
    assert first.day_linear_rgb == pytest.approx(
        (0.2862094288764995, 0.46422913639396673, 0.15768364281664943),
        abs=1e-12,
    )
    assert first.triggered_linear_rgb == pytest.approx(
        (0.29541162263806936, 0.4986007036513035, 0.17100682996040184),
        abs=1e-12,
    )
    assert np.all(np.asarray(first.triggered_linear_rgb) > first.day_linear_rgb)
    assert first.day_saturated_channels == (False, False, False)
    assert first.triggered_saturated_channels == (False, False, False)
    assert first.calibration_sha256 == calibration.canonical_sha256
    assert first.calibration_source_sha256 == calibration.source_sha256
    assert first.calibration_schema_version == calibration.schema_version
    assert first.calibration_id == calibration.provenance.calibration_id


def test_transport_supports_numpy_124_trapezoid_api(monkeypatch):
    calibration = load_fluorescence_calibration(SYNTHETIC_FIXTURE)
    expected = compute_fluorescence_transport(calibration, seed=42)

    had_modern_api = hasattr(np, "trapezoid")
    monkeypatch.delattr(np, "trapezoid", raising=False)
    if had_modern_api:
        with pytest.warns(DeprecationWarning):
            legacy_api_result = compute_fluorescence_transport(calibration, seed=42)
    else:
        legacy_api_result = compute_fluorescence_transport(calibration, seed=42)

    assert legacy_api_result.day_linear_rgb == pytest.approx(expected.day_linear_rgb)
    assert legacy_api_result.triggered_linear_rgb == pytest.approx(
        expected.triggered_linear_rgb
    )


def test_canonical_hash_ignores_json_formatting_but_source_hash_does_not(tmp_path):
    document = _fixture_document()
    reordered = tmp_path / "reordered.json"
    reordered.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    original = load_fluorescence_calibration(SYNTHETIC_FIXTURE)
    reformatted = load_fluorescence_calibration(reordered)

    assert original.canonical_sha256 == reformatted.canonical_sha256
    assert original.source_sha256 != reformatted.source_sha256


def test_direct_measured_emission_response_model_is_supported(tmp_path):
    document = _fixture_document()
    document["material"]["fluorescence_model"] = {
        "type": "measured_emission_response",
        "emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv": [
            0,
            0,
            0.0001,
            0.001,
            0.002,
            0.001,
            0.0001,
            0,
        ],
    }
    # This model does not use excitation efficiency, but the versioned uncertainty
    # contract still requires an explicit fixed bound rather than supplying a default.
    document["uncertainty"]["material_batch"]["excitation_efficiency_scale"] = {
        "low": 1,
        "high": 1,
    }
    calibration = load_fluorescence_calibration(_write_document(tmp_path, document))
    result = compute_fluorescence_transport(calibration, seed=42)

    assert isinstance(
        calibration.material.fluorescence_model, MeasuredEmissionResponseModel
    )
    assert np.all(np.isfinite(result.triggered_linear_rgb_unclipped))
    assert np.any(
        np.asarray(result.triggered_linear_rgb_unclipped)
        > np.asarray(result.day_linear_rgb_unclipped)
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "2.0.0", "Unsupported schema_version"),
        ("calibration_type", "unspecified", "calibration_type must be one of"),
    ],
)
def test_rejects_unsupported_contract_values(tmp_path, field, value, match):
    document = _fixture_document()
    document[field] = value

    with pytest.raises(CalibrationError, match=match):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_rejects_wrong_units_instead_of_guessing(tmp_path):
    document = _fixture_document()
    document["units"]["wavelength"] = "um"

    with pytest.raises(CalibrationError, match="units.wavelength must be exactly"):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_rejects_mismatched_and_nonmonotonic_spectral_grids(tmp_path):
    mismatched = _fixture_document()
    mismatched["ambient_illuminant"]["spectral_irradiance_w_m2_nm"].pop()
    with pytest.raises(CalibrationError, match="does not match wavelength grid length"):
        load_fluorescence_calibration(
            _write_document(tmp_path, mismatched, "mismatched.json")
        )

    nonmonotonic = _fixture_document()
    nonmonotonic["wavelength_grid"]["values_nm"][3] = 400
    with pytest.raises(CalibrationError, match="strictly increasing"):
        load_fluorescence_calibration(
            _write_document(tmp_path, nonmonotonic, "nonmonotonic.json")
        )


def test_rejects_missing_uncertainty_instead_of_inventing_a_default(tmp_path):
    document = _fixture_document()
    del document["uncertainty"]["irradiance"]["uv_scale"]

    with pytest.raises(CalibrationError, match="missing .*uv_scale"):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_rejects_bounds_that_can_leave_a_physical_range(tmp_path):
    document = _fixture_document()
    document["uncertainty"]["material_batch"]["substrate_reflectance_scale"] = {
        "low": 1,
        "high": 1.3,
    }

    with pytest.raises(CalibrationError, match="push reflectance above 1"):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_synthetic_fixture_requires_a_non_empirical_notice(tmp_path):
    document = _fixture_document()
    document["provenance"]["non_empirical_notice"] = ""

    with pytest.raises(CalibrationError, match="require a prominent"):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_measured_calibration_requires_empirical_provenance(tmp_path):
    document = _fixture_document()
    document["calibration_type"] = "measured"

    with pytest.raises(CalibrationError, match="must set provenance.is_empirical=true"):
        load_fluorescence_calibration(_write_document(tmp_path, document))


def test_rejects_duplicate_keys_and_nonfinite_json(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        encoding="utf-8",
    )
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(CalibrationError, match="duplicate key"):
        load_fluorescence_calibration(duplicate)
    with pytest.raises(CalibrationError, match="non-finite numeric constant"):
        load_fluorescence_calibration(nonfinite)


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_seed_must_be_an_integer(seed):
    calibration = load_fluorescence_calibration(SYNTHETIC_FIXTURE)

    with pytest.raises(TypeError, match="seed must be an integer"):
        compute_fluorescence_transport(calibration, seed=seed)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_seed_must_fit_uint64(seed):
    calibration = load_fluorescence_calibration(SYNTHETIC_FIXTURE)

    with pytest.raises(ValueError, match="seed must be in"):
        compute_fluorescence_transport(calibration, seed=seed)


def test_fixture_document_is_not_mutated_by_loader(tmp_path):
    document = _fixture_document()
    before = copy.deepcopy(document)

    load_fluorescence_calibration(_write_document(tmp_path, document))

    assert document == before
