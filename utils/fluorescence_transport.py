"""Measured spectral transport for fluorescent traffic-sign materials.

This module deliberately has no built-in material, illuminant, or camera defaults.
Every physical quantity and every uncertainty bound must come from a validated,
versioned calibration document.  The included repository fixture declares itself
synthetic and is suitable only for tests and API examples.

The transport is a compact Lambertian radiometric model, not a replacement for a
full optical renderer.  On one shared wavelength grid it computes

* reflected day radiance from ambient irradiance, substrate reflectance, and the
  coating's day transmittance;
* reflected triggered radiance from ambient plus trigger irradiance;
* fluorescent radiance from either an excitation/emission model or a directly
  measured per-wavelength emission response; and
* relative camera raw RGB followed by a calibrated linear 3x3 ISP transform.

No gamma encoding is performed.  Results expose both unclipped and [0, 1]-clipped
linear RGB so saturation is never hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np


SCHEMA_VERSION = "1.0.0"
SCHEMA_FILENAME = "fluorescence_transport_v1.schema.json"
CALIBRATION_TYPES = frozenset({"measured", "synthetic_non_empirical"})
EXPECTED_UNITS = {
    "wavelength": "nm",
    "spectral_irradiance": "W m^-2 nm^-1",
    "spectral_radiance": "W m^-2 sr^-1 nm^-1",
    "camera_sensitivity": "relative",
    "linear_rgb": "relative",
}


class CalibrationError(ValueError):
    """Raised when a spectral calibration document is invalid or incomplete."""


def fluorescence_transport_schema_path() -> Path:
    """Return the repository path for the JSON Schema matching this loader."""

    return Path(__file__).resolve().parents[1] / "schemas" / SCHEMA_FILENAME


def _trapezoidal_integral(
    values: np.ndarray, wavelength: np.ndarray, *, axis: int = -1
) -> Any:
    """Integrate on NumPy 1.24+ without changing the declared algorithm."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, wavelength, axis=axis)
    legacy_trapz = getattr(np, "trapz")
    return legacy_trapz(values, wavelength, axis=axis)


@dataclass(frozen=True)
class ScalarBounds:
    low: float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Provenance:
    calibration_id: str
    description: str
    source: str
    is_empirical: bool
    non_empirical_notice: str


@dataclass(frozen=True)
class ExcitationEmissionModel:
    excitation_efficiency: Tuple[float, ...]
    emission_relative_spectrum: Tuple[float, ...]
    radiant_efficiency: float


@dataclass(frozen=True)
class MeasuredEmissionResponseModel:
    emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv: Tuple[float, ...]


FluorescenceModel = Union[ExcitationEmissionModel, MeasuredEmissionResponseModel]


@dataclass(frozen=True)
class MaterialCalibration:
    name: str
    day_spectral_transmittance: Tuple[float, ...]
    fluorescence_model: FluorescenceModel


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    rgb_sensitivity: Tuple[Tuple[float, ...], ...]
    exposure_scale_m2_sr_per_w: float
    isp_matrix_raw_to_linear_rgb: Tuple[Tuple[float, ...], ...]
    linear_rgb_offset: Tuple[float, ...]


@dataclass(frozen=True)
class MaterialUncertainty:
    substrate_reflectance_scale: ScalarBounds
    day_transmittance_scale: ScalarBounds
    excitation_efficiency_scale: ScalarBounds
    fluorescence_output_scale: ScalarBounds


@dataclass(frozen=True)
class CameraUncertainty:
    rgb_sensitivity_channel_scale: Tuple[ScalarBounds, ...]
    exposure_scale: ScalarBounds
    isp_output_channel_scale: Tuple[ScalarBounds, ...]


@dataclass(frozen=True)
class IrradianceUncertainty:
    ambient_scale: ScalarBounds
    uv_scale: ScalarBounds


@dataclass(frozen=True)
class UncertaintyCalibration:
    sampling: str
    material_batch: MaterialUncertainty
    camera: CameraUncertainty
    irradiance: IrradianceUncertainty


@dataclass(frozen=True)
class FluorescenceCalibration:
    """Validated, immutable calibration loaded from one JSON document."""

    schema_version: str
    calibration_type: str
    provenance: Provenance
    wavelength_nm: Tuple[float, ...]
    ambient_illuminant_name: str
    ambient_spectral_irradiance_w_m2_nm: Tuple[float, ...]
    uv_trigger_name: str
    uv_spectral_irradiance_w_m2_nm: Tuple[float, ...]
    substrate_name: str
    substrate_spectral_reflectance: Tuple[float, ...]
    material: MaterialCalibration
    camera: CameraCalibration
    uncertainty: UncertaintyCalibration
    canonical_sha256: str
    source_sha256: str
    source_path: str


@dataclass(frozen=True)
class UncertaintyDraw:
    material_batch: Mapping[str, Any]
    camera: Mapping[str, Any]
    irradiance: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_batch": dict(self.material_batch),
            "camera": dict(self.camera),
            "irradiance": dict(self.irradiance),
        }


@dataclass(frozen=True)
class TransportResult:
    seed: int
    calibration_schema_version: str
    calibration_id: str
    calibration_sha256: str
    calibration_source_sha256: str
    calibration_type: str
    day_raw_rgb: Tuple[float, float, float]
    triggered_raw_rgb: Tuple[float, float, float]
    day_linear_rgb_unclipped: Tuple[float, float, float]
    triggered_linear_rgb_unclipped: Tuple[float, float, float]
    day_linear_rgb: Tuple[float, float, float]
    triggered_linear_rgb: Tuple[float, float, float]
    day_saturated_channels: Tuple[bool, bool, bool]
    triggered_saturated_channels: Tuple[bool, bool, bool]
    uncertainty_draw: UncertaintyDraw

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "calibration_schema_version": self.calibration_schema_version,
            "calibration_id": self.calibration_id,
            "calibration_sha256": self.calibration_sha256,
            "calibration_source_sha256": self.calibration_source_sha256,
            "calibration_type": self.calibration_type,
            "day_raw_rgb": list(self.day_raw_rgb),
            "triggered_raw_rgb": list(self.triggered_raw_rgb),
            "day_linear_rgb_unclipped": list(self.day_linear_rgb_unclipped),
            "triggered_linear_rgb_unclipped": list(
                self.triggered_linear_rgb_unclipped
            ),
            "day_linear_rgb": list(self.day_linear_rgb),
            "triggered_linear_rgb": list(self.triggered_linear_rgb),
            "day_saturated_channels": list(self.day_saturated_channels),
            "triggered_saturated_channels": list(
                self.triggered_saturated_channels
            ),
            "uncertainty_draw": self.uncertainty_draw.to_dict(),
        }


def _reject_json_constant(value: str) -> None:
    raise CalibrationError(f"JSON contains non-finite numeric constant {value!r}")


def _object_without_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _require_object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CalibrationError(f"{path} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, required: Sequence[str], path: str
) -> None:
    required_set = set(required)
    actual = set(value)
    missing = sorted(required_set - actual)
    extra = sorted(actual - required_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise CalibrationError(f"{path} has invalid fields: {', '.join(details)}")


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{path} must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{path} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{path} must be finite")
    return result


def _number_vector(
    value: Any,
    path: str,
    *,
    length: int | None = None,
    low: float | None = None,
    high: float | None = None,
) -> Tuple[float, ...]:
    if not isinstance(value, list):
        raise CalibrationError(f"{path} must be a JSON array")
    if length is not None and len(value) != length:
        raise CalibrationError(
            f"{path} length {len(value)} does not match wavelength grid length {length}"
        )
    result = tuple(
        _number(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    for index, item in enumerate(result):
        if low is not None and item < low:
            raise CalibrationError(f"{path}[{index}] must be >= {low}")
        if high is not None and item > high:
            raise CalibrationError(f"{path}[{index}] must be <= {high}")
    return result


def _matrix(
    value: Any,
    path: str,
    *,
    rows: int,
    columns: int,
    low: float | None = None,
    high: float | None = None,
) -> Tuple[Tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != rows:
        raise CalibrationError(f"{path} must contain exactly {rows} rows")
    return tuple(
        _number_vector(
            row,
            f"{path}[{index}]",
            length=columns,
            low=low,
            high=high,
        )
        for index, row in enumerate(value)
    )


def _bounds(value: Any, path: str) -> ScalarBounds:
    obj = _require_object(value, path)
    _require_exact_keys(obj, required=("low", "high"), path=path)
    low = _number(obj["low"], f"{path}.low")
    high = _number(obj["high"], f"{path}.high")
    if low < 0:
        raise CalibrationError(f"{path}.low must be nonnegative")
    if high < low:
        raise CalibrationError(f"{path}.high must be >= {path}.low")
    return ScalarBounds(low=low, high=high)


def _bounds_vector(value: Any, path: str, *, length: int) -> Tuple[ScalarBounds, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise CalibrationError(f"{path} must contain exactly {length} bounds objects")
    return tuple(_bounds(item, f"{path}[{index}]") for index, item in enumerate(value))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(document: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def load_fluorescence_calibration(
    path: Union[str, Path],
) -> FluorescenceCalibration:
    """Load and strictly validate a version 1 fluorescence calibration.

    The loader does not interpolate grids, infer units, fill missing values, or
    supply uncertainty defaults.  Unknown fields are rejected so misspellings do
    not silently alter the physical model.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise CalibrationError(f"Calibration file does not exist: {source}")
    payload = source.read_bytes()
    try:
        document = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except UnicodeDecodeError as exc:
        raise CalibrationError(f"Calibration JSON must be UTF-8: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"Invalid calibration JSON at {source}: {exc}") from exc

    root = _require_object(document, "root")
    _require_exact_keys(
        root,
        required=(
            "schema_version",
            "calibration_type",
            "provenance",
            "units",
            "wavelength_grid",
            "ambient_illuminant",
            "uv_trigger",
            "substrate",
            "material",
            "camera",
            "uncertainty",
        ),
        path="root",
    )

    schema_version = _require_string(root["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise CalibrationError(
            f"Unsupported schema_version {schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    calibration_type = _require_string(root["calibration_type"], "calibration_type")
    if calibration_type not in CALIBRATION_TYPES:
        raise CalibrationError(
            f"calibration_type must be one of {sorted(CALIBRATION_TYPES)}"
        )

    provenance_obj = _require_object(root["provenance"], "provenance")
    _require_exact_keys(
        provenance_obj,
        required=(
            "calibration_id",
            "description",
            "source",
            "is_empirical",
            "non_empirical_notice",
        ),
        path="provenance",
    )
    is_empirical = provenance_obj["is_empirical"]
    if not isinstance(is_empirical, bool):
        raise CalibrationError("provenance.is_empirical must be a JSON boolean")
    notice = provenance_obj["non_empirical_notice"]
    if not isinstance(notice, str):
        raise CalibrationError("provenance.non_empirical_notice must be a string")
    if calibration_type == "measured" and not is_empirical:
        raise CalibrationError(
            "measured calibrations must set provenance.is_empirical=true"
        )
    if calibration_type == "synthetic_non_empirical":
        if is_empirical:
            raise CalibrationError(
                "synthetic_non_empirical calibrations must set "
                "provenance.is_empirical=false"
            )
        if not notice.strip():
            raise CalibrationError(
                "synthetic_non_empirical calibrations require a prominent "
                "provenance.non_empirical_notice"
            )
    provenance = Provenance(
        calibration_id=_require_string(
            provenance_obj["calibration_id"], "provenance.calibration_id"
        ),
        description=_require_string(
            provenance_obj["description"], "provenance.description"
        ),
        source=_require_string(provenance_obj["source"], "provenance.source"),
        is_empirical=is_empirical,
        non_empirical_notice=notice,
    )

    units = _require_object(root["units"], "units")
    _require_exact_keys(units, required=tuple(EXPECTED_UNITS), path="units")
    for name, expected in EXPECTED_UNITS.items():
        actual = units[name]
        if actual != expected:
            raise CalibrationError(
                f"units.{name} must be exactly {expected!r}, got {actual!r}"
            )

    grid_obj = _require_object(root["wavelength_grid"], "wavelength_grid")
    _require_exact_keys(
        grid_obj, required=("values_nm", "integration"), path="wavelength_grid"
    )
    if grid_obj["integration"] != "trapezoidal":
        raise CalibrationError("wavelength_grid.integration must be 'trapezoidal'")
    wavelength_nm = _number_vector(
        grid_obj["values_nm"], "wavelength_grid.values_nm", low=0.0
    )
    if len(wavelength_nm) < 2:
        raise CalibrationError("wavelength_grid.values_nm needs at least two samples")
    if any(right <= left for left, right in zip(wavelength_nm, wavelength_nm[1:])):
        raise CalibrationError("wavelength_grid.values_nm must be strictly increasing")
    spectral_length = len(wavelength_nm)

    ambient_obj = _require_object(root["ambient_illuminant"], "ambient_illuminant")
    _require_exact_keys(
        ambient_obj,
        required=("name", "spectral_irradiance_w_m2_nm"),
        path="ambient_illuminant",
    )
    ambient_illuminant_name = _require_string(
        ambient_obj["name"], "ambient_illuminant.name"
    )
    ambient = _number_vector(
        ambient_obj["spectral_irradiance_w_m2_nm"],
        "ambient_illuminant.spectral_irradiance_w_m2_nm",
        length=spectral_length,
        low=0.0,
    )

    uv_obj = _require_object(root["uv_trigger"], "uv_trigger")
    _require_exact_keys(
        uv_obj,
        required=("name", "spectral_irradiance_w_m2_nm"),
        path="uv_trigger",
    )
    uv_trigger_name = _require_string(uv_obj["name"], "uv_trigger.name")
    uv = _number_vector(
        uv_obj["spectral_irradiance_w_m2_nm"],
        "uv_trigger.spectral_irradiance_w_m2_nm",
        length=spectral_length,
        low=0.0,
    )

    wavelengths_array = np.asarray(wavelength_nm, dtype=np.float64)
    if float(_trapezoidal_integral(np.asarray(ambient), wavelengths_array)) <= 0:
        raise CalibrationError(
            "ambient_illuminant must have positive integrated irradiance"
        )
    if float(_trapezoidal_integral(np.asarray(uv), wavelengths_array)) <= 0:
        raise CalibrationError("uv_trigger must have positive integrated irradiance")

    substrate_obj = _require_object(root["substrate"], "substrate")
    _require_exact_keys(
        substrate_obj,
        required=("name", "spectral_reflectance"),
        path="substrate",
    )
    substrate_name = _require_string(substrate_obj["name"], "substrate.name")
    substrate_reflectance = _number_vector(
        substrate_obj["spectral_reflectance"],
        "substrate.spectral_reflectance",
        length=spectral_length,
        low=0.0,
        high=1.0,
    )

    material_obj = _require_object(root["material"], "material")
    _require_exact_keys(
        material_obj,
        required=("name", "day_spectral_transmittance", "fluorescence_model"),
        path="material",
    )
    day_transmittance = _number_vector(
        material_obj["day_spectral_transmittance"],
        "material.day_spectral_transmittance",
        length=spectral_length,
        low=0.0,
        high=1.0,
    )
    fluorescence_obj = _require_object(
        material_obj["fluorescence_model"], "material.fluorescence_model"
    )
    model_type = fluorescence_obj.get("type")
    if model_type == "excitation_emission":
        _require_exact_keys(
            fluorescence_obj,
            required=(
                "type",
                "excitation_efficiency",
                "emission_relative_spectrum",
                "radiant_efficiency",
            ),
            path="material.fluorescence_model",
        )
        excitation = _number_vector(
            fluorescence_obj["excitation_efficiency"],
            "material.fluorescence_model.excitation_efficiency",
            length=spectral_length,
            low=0.0,
            high=1.0,
        )
        emission = _number_vector(
            fluorescence_obj["emission_relative_spectrum"],
            "material.fluorescence_model.emission_relative_spectrum",
            length=spectral_length,
            low=0.0,
        )
        if float(_trapezoidal_integral(np.asarray(emission), wavelengths_array)) <= 0:
            raise CalibrationError(
                "material fluorescence emission spectrum must have positive area"
            )
        radiant_efficiency = _number(
            fluorescence_obj["radiant_efficiency"],
            "material.fluorescence_model.radiant_efficiency",
        )
        if not 0 <= radiant_efficiency <= 1:
            raise CalibrationError(
                "material.fluorescence_model.radiant_efficiency must be in [0, 1]"
            )
        fluorescence_model: FluorescenceModel = ExcitationEmissionModel(
            excitation_efficiency=excitation,
            emission_relative_spectrum=emission,
            radiant_efficiency=radiant_efficiency,
        )
    elif model_type == "measured_emission_response":
        _require_exact_keys(
            fluorescence_obj,
            required=(
                "type",
                "emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv",
            ),
            path="material.fluorescence_model",
        )
        response = _number_vector(
            fluorescence_obj[
                "emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv"
            ],
            "material.fluorescence_model."
            "emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv",
            length=spectral_length,
            low=0.0,
        )
        if float(_trapezoidal_integral(np.asarray(response), wavelengths_array)) <= 0:
            raise CalibrationError(
                "measured fluorescence emission response must have positive area"
            )
        fluorescence_model = MeasuredEmissionResponseModel(
            emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv=response
        )
    else:
        raise CalibrationError(
            "material.fluorescence_model.type must be 'excitation_emission' or "
            "'measured_emission_response'"
        )

    material = MaterialCalibration(
        name=_require_string(material_obj["name"], "material.name"),
        day_spectral_transmittance=day_transmittance,
        fluorescence_model=fluorescence_model,
    )

    camera_obj = _require_object(root["camera"], "camera")
    _require_exact_keys(
        camera_obj,
        required=(
            "name",
            "rgb_sensitivity",
            "exposure_scale_m2_sr_per_w",
            "isp_matrix_raw_to_linear_rgb",
            "linear_rgb_offset",
        ),
        path="camera",
    )
    rgb_sensitivity = _matrix(
        camera_obj["rgb_sensitivity"],
        "camera.rgb_sensitivity",
        rows=3,
        columns=spectral_length,
        low=0.0,
        high=1.0,
    )
    for channel_index, channel in enumerate(rgb_sensitivity):
        if float(_trapezoidal_integral(np.asarray(channel), wavelengths_array)) <= 0:
            raise CalibrationError(
                f"camera.rgb_sensitivity[{channel_index}] must have positive area"
            )
    exposure_scale = _number(
        camera_obj["exposure_scale_m2_sr_per_w"],
        "camera.exposure_scale_m2_sr_per_w",
    )
    if exposure_scale <= 0:
        raise CalibrationError("camera.exposure_scale_m2_sr_per_w must be positive")
    isp_matrix = _matrix(
        camera_obj["isp_matrix_raw_to_linear_rgb"],
        "camera.isp_matrix_raw_to_linear_rgb",
        rows=3,
        columns=3,
    )
    linear_rgb_offset = _number_vector(
        camera_obj["linear_rgb_offset"], "camera.linear_rgb_offset", length=3
    )
    camera = CameraCalibration(
        name=_require_string(camera_obj["name"], "camera.name"),
        rgb_sensitivity=rgb_sensitivity,
        exposure_scale_m2_sr_per_w=exposure_scale,
        isp_matrix_raw_to_linear_rgb=isp_matrix,
        linear_rgb_offset=linear_rgb_offset,
    )

    uncertainty_obj = _require_object(root["uncertainty"], "uncertainty")
    _require_exact_keys(
        uncertainty_obj,
        required=("sampling", "material_batch", "camera", "irradiance"),
        path="uncertainty",
    )
    if uncertainty_obj["sampling"] != "independent_uniform":
        raise CalibrationError("uncertainty.sampling must be 'independent_uniform'")

    material_uncertainty_obj = _require_object(
        uncertainty_obj["material_batch"], "uncertainty.material_batch"
    )
    _require_exact_keys(
        material_uncertainty_obj,
        required=(
            "substrate_reflectance_scale",
            "day_transmittance_scale",
            "excitation_efficiency_scale",
            "fluorescence_output_scale",
        ),
        path="uncertainty.material_batch",
    )
    material_uncertainty = MaterialUncertainty(
        substrate_reflectance_scale=_bounds(
            material_uncertainty_obj["substrate_reflectance_scale"],
            "uncertainty.material_batch.substrate_reflectance_scale",
        ),
        day_transmittance_scale=_bounds(
            material_uncertainty_obj["day_transmittance_scale"],
            "uncertainty.material_batch.day_transmittance_scale",
        ),
        excitation_efficiency_scale=_bounds(
            material_uncertainty_obj["excitation_efficiency_scale"],
            "uncertainty.material_batch.excitation_efficiency_scale",
        ),
        fluorescence_output_scale=_bounds(
            material_uncertainty_obj["fluorescence_output_scale"],
            "uncertainty.material_batch.fluorescence_output_scale",
        ),
    )

    camera_uncertainty_obj = _require_object(
        uncertainty_obj["camera"], "uncertainty.camera"
    )
    _require_exact_keys(
        camera_uncertainty_obj,
        required=(
            "rgb_sensitivity_channel_scale",
            "exposure_scale",
            "isp_output_channel_scale",
        ),
        path="uncertainty.camera",
    )
    camera_uncertainty = CameraUncertainty(
        rgb_sensitivity_channel_scale=_bounds_vector(
            camera_uncertainty_obj["rgb_sensitivity_channel_scale"],
            "uncertainty.camera.rgb_sensitivity_channel_scale",
            length=3,
        ),
        exposure_scale=_bounds(
            camera_uncertainty_obj["exposure_scale"],
            "uncertainty.camera.exposure_scale",
        ),
        isp_output_channel_scale=_bounds_vector(
            camera_uncertainty_obj["isp_output_channel_scale"],
            "uncertainty.camera.isp_output_channel_scale",
            length=3,
        ),
    )

    irradiance_uncertainty_obj = _require_object(
        uncertainty_obj["irradiance"], "uncertainty.irradiance"
    )
    _require_exact_keys(
        irradiance_uncertainty_obj,
        required=("ambient_scale", "uv_scale"),
        path="uncertainty.irradiance",
    )
    irradiance_uncertainty = IrradianceUncertainty(
        ambient_scale=_bounds(
            irradiance_uncertainty_obj["ambient_scale"],
            "uncertainty.irradiance.ambient_scale",
        ),
        uv_scale=_bounds(
            irradiance_uncertainty_obj["uv_scale"],
            "uncertainty.irradiance.uv_scale",
        ),
    )
    uncertainty = UncertaintyCalibration(
        sampling="independent_uniform",
        material_batch=material_uncertainty,
        camera=camera_uncertainty,
        irradiance=irradiance_uncertainty,
    )

    # Bounded draws must preserve quantities whose declared physical range is [0, 1].
    if (
        max(substrate_reflectance)
        * material_uncertainty.substrate_reflectance_scale.high
        > 1
    ):
        raise CalibrationError(
            "substrate_reflectance_scale upper bound can push reflectance above 1"
        )
    if max(day_transmittance) * material_uncertainty.day_transmittance_scale.high > 1:
        raise CalibrationError(
            "day_transmittance_scale upper bound can push transmittance above 1"
        )
    if isinstance(fluorescence_model, ExcitationEmissionModel):
        if (
            max(fluorescence_model.excitation_efficiency)
            * material_uncertainty.excitation_efficiency_scale.high
            > 1
        ):
            raise CalibrationError(
                "excitation_efficiency_scale upper bound can push efficiency above 1"
            )
        if (
            fluorescence_model.radiant_efficiency
            * material_uncertainty.fluorescence_output_scale.high
            > 1
        ):
            raise CalibrationError(
                "fluorescence_output_scale upper bound can push radiant "
                "efficiency above 1"
            )
    for channel_index, (channel, bound) in enumerate(
        zip(rgb_sensitivity, camera_uncertainty.rgb_sensitivity_channel_scale)
    ):
        if max(channel) * bound.high > 1:
            raise CalibrationError(
                "camera sensitivity uncertainty can push channel "
                f"{channel_index} sensitivity above 1"
            )

    return FluorescenceCalibration(
        schema_version=schema_version,
        calibration_type=calibration_type,
        provenance=provenance,
        wavelength_nm=wavelength_nm,
        ambient_illuminant_name=ambient_illuminant_name,
        ambient_spectral_irradiance_w_m2_nm=ambient,
        uv_trigger_name=uv_trigger_name,
        uv_spectral_irradiance_w_m2_nm=uv,
        substrate_name=substrate_name,
        substrate_spectral_reflectance=substrate_reflectance,
        material=material,
        camera=camera,
        uncertainty=uncertainty,
        canonical_sha256=_canonical_hash(root),
        source_sha256=_sha256_bytes(payload),
        source_path=str(source),
    )


def _draw_uncertainty(
    calibration: FluorescenceCalibration, *, seed: int
) -> UncertaintyDraw:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > np.iinfo(np.uint64).max:
        raise ValueError("seed must be in [0, 2**64 - 1]")
    rng = np.random.default_rng(seed)
    material = calibration.uncertainty.material_batch
    camera = calibration.uncertainty.camera
    irradiance = calibration.uncertainty.irradiance
    return UncertaintyDraw(
        material_batch={
            "substrate_reflectance_scale": (
                material.substrate_reflectance_scale.sample(rng)
            ),
            "day_transmittance_scale": material.day_transmittance_scale.sample(rng),
            "excitation_efficiency_scale": (
                material.excitation_efficiency_scale.sample(rng)
            ),
            "fluorescence_output_scale": material.fluorescence_output_scale.sample(rng),
        },
        camera={
            "rgb_sensitivity_channel_scale": [
                bound.sample(rng) for bound in camera.rgb_sensitivity_channel_scale
            ],
            "exposure_scale": camera.exposure_scale.sample(rng),
            "isp_output_channel_scale": [
                bound.sample(rng) for bound in camera.isp_output_channel_scale
            ],
        },
        irradiance={
            "ambient_scale": irradiance.ambient_scale.sample(rng),
            "uv_scale": irradiance.uv_scale.sample(rng),
        },
    )


def compute_fluorescence_transport(
    calibration: FluorescenceCalibration, *, seed: int
) -> TransportResult:
    """Compute day and UV-triggered linear RGB for one bounded uncertainty draw."""

    if not isinstance(calibration, FluorescenceCalibration):
        raise TypeError("calibration must be loaded by load_fluorescence_calibration")
    draw = _draw_uncertainty(calibration, seed=seed)

    wavelength = np.asarray(calibration.wavelength_nm, dtype=np.float64)
    ambient = np.asarray(
        calibration.ambient_spectral_irradiance_w_m2_nm, dtype=np.float64
    ) * float(draw.irradiance["ambient_scale"])
    uv = np.asarray(
        calibration.uv_spectral_irradiance_w_m2_nm, dtype=np.float64
    ) * float(draw.irradiance["uv_scale"])
    substrate = np.asarray(
        calibration.substrate_spectral_reflectance, dtype=np.float64
    ) * float(draw.material_batch["substrate_reflectance_scale"])
    transmittance = np.asarray(
        calibration.material.day_spectral_transmittance, dtype=np.float64
    ) * float(draw.material_batch["day_transmittance_scale"])

    day_radiance = ambient * substrate * transmittance / math.pi
    triggered_radiance = (ambient + uv) * substrate * transmittance / math.pi

    model = calibration.material.fluorescence_model
    fluorescence_output_scale = float(
        draw.material_batch["fluorescence_output_scale"]
    )
    if isinstance(model, ExcitationEmissionModel):
        excitation = np.asarray(model.excitation_efficiency, dtype=np.float64) * float(
            draw.material_batch["excitation_efficiency_scale"]
        )
        absorbed_excitation_w_m2 = float(
            _trapezoidal_integral(uv * excitation, wavelength)
        )
        relative_emission = np.asarray(
            model.emission_relative_spectrum, dtype=np.float64
        )
        normalized_emission_per_nm = relative_emission / float(
            _trapezoidal_integral(relative_emission, wavelength)
        )
        fluorescent_radiance = (
            absorbed_excitation_w_m2
            * model.radiant_efficiency
            * fluorescence_output_scale
            * normalized_emission_per_nm
            / math.pi
        )
    else:
        integrated_uv_w_m2 = float(_trapezoidal_integral(uv, wavelength))
        fluorescent_radiance = (
            integrated_uv_w_m2
            * np.asarray(
                model.emission_radiance_response_per_uv_irradiance_sr_inv_nm_inv,
                dtype=np.float64,
            )
            * fluorescence_output_scale
        )
    triggered_radiance = triggered_radiance + fluorescent_radiance

    sensitivity = np.asarray(calibration.camera.rgb_sensitivity, dtype=np.float64)
    sensitivity_scales = np.asarray(
        draw.camera["rgb_sensitivity_channel_scale"], dtype=np.float64
    )
    sensitivity = sensitivity * sensitivity_scales[:, None]
    exposure = calibration.camera.exposure_scale_m2_sr_per_w * float(
        draw.camera["exposure_scale"]
    )

    day_raw = exposure * _trapezoidal_integral(
        sensitivity * day_radiance[None, :], wavelength, axis=1
    )
    triggered_raw = exposure * _trapezoidal_integral(
        sensitivity * triggered_radiance[None, :], wavelength, axis=1
    )

    isp = np.asarray(
        calibration.camera.isp_matrix_raw_to_linear_rgb, dtype=np.float64
    )
    isp_output_scales = np.asarray(
        draw.camera["isp_output_channel_scale"], dtype=np.float64
    )
    isp = isp_output_scales[:, None] * isp
    offset = np.asarray(calibration.camera.linear_rgb_offset, dtype=np.float64)
    day_unclipped = isp @ day_raw + offset
    triggered_unclipped = isp @ triggered_raw + offset
    day_clipped = np.clip(day_unclipped, 0.0, 1.0)
    triggered_clipped = np.clip(triggered_unclipped, 0.0, 1.0)

    def triple(values: np.ndarray) -> Tuple[float, float, float]:
        return tuple(float(value) for value in values)  # type: ignore[return-value]

    def saturation(values: np.ndarray) -> Tuple[bool, bool, bool]:
        return tuple(  # type: ignore[return-value]
            bool(value < 0 or value > 1) for value in values
        )

    return TransportResult(
        seed=seed,
        calibration_schema_version=calibration.schema_version,
        calibration_id=calibration.provenance.calibration_id,
        calibration_sha256=calibration.canonical_sha256,
        calibration_source_sha256=calibration.source_sha256,
        calibration_type=calibration.calibration_type,
        day_raw_rgb=triple(day_raw),
        triggered_raw_rgb=triple(triggered_raw),
        day_linear_rgb_unclipped=triple(day_unclipped),
        triggered_linear_rgb_unclipped=triple(triggered_unclipped),
        day_linear_rgb=triple(day_clipped),
        triggered_linear_rgb=triple(triggered_clipped),
        day_saturated_channels=saturation(day_unclipped),
        triggered_saturated_channels=saturation(triggered_unclipped),
        uncertainty_draw=draw,
    )


__all__ = [
    "CALIBRATION_TYPES",
    "EXPECTED_UNITS",
    "SCHEMA_FILENAME",
    "SCHEMA_VERSION",
    "CalibrationError",
    "CameraCalibration",
    "ExcitationEmissionModel",
    "FluorescenceCalibration",
    "MaterialCalibration",
    "MeasuredEmissionResponseModel",
    "Provenance",
    "TransportResult",
    "UncertaintyDraw",
    "compute_fluorescence_transport",
    "fluorescence_transport_schema_path",
    "load_fluorescence_calibration",
]
