"""Traffic-sign asset profiles shared by training, evaluation, and baselines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from detectors.class_names import ClassReference


@dataclass(frozen=True)
class SignAssets:
    profile: str
    day_image: Path
    active_image: Path
    source_class: ClassReference


_PROFILE_DEFAULTS = {
    "stop": ("stop_sign.png", "stop_sign_uv.png", "stop sign"),
    "speed_limit": ("speed_limit_sign.png", "speed_limit_sign_uv.png", "speed limit sign"),
}


def resolve_sign_assets(
    *,
    data_dir: Union[str, Path],
    profile: str = "stop",
    sign_image: Optional[Union[str, Path]] = None,
    sign_active_image: Optional[Union[str, Path]] = None,
    source_class: Optional[ClassReference] = None,
) -> SignAssets:
    """Resolve a built-in or custom sign profile and validate its day asset.

    The active/UV image is optional: when it is absent, callers receive the day
    image path and can reuse the day pixels.  A speed-limit run generally needs
    custom detector weights because common COCO checkpoints do not expose a
    speed-limit-sign class.
    """
    key = str(profile or "stop").strip().lower().replace("-", "_")
    if key == "speed":
        key = "speed_limit"
    if key not in (*_PROFILE_DEFAULTS.keys(), "custom"):
        choices = ", ".join([*_PROFILE_DEFAULTS.keys(), "custom"])
        raise ValueError(f"Unknown sign profile {profile!r}; choose one of: {choices}")

    root = Path(data_dir)
    if key == "custom":
        if not sign_image:
            raise ValueError("--sign-image is required when --sign-profile custom is used")
        if source_class is None or not str(source_class).strip():
            raise ValueError("--source-class is required when --sign-profile custom is used")
        default_day = str(sign_image)
        default_active = str(sign_active_image or sign_image)
        default_source: ClassReference = source_class
    else:
        default_day_name, default_active_name, default_source = _PROFILE_DEFAULTS[key]
        default_day = str(sign_image or (root / default_day_name))
        default_active = str(
            sign_active_image
            or (sign_image if sign_image is not None else (root / default_active_name))
        )

    day_path = Path(default_day)
    if not day_path.is_file():
        profile_hint = (
            " Supply --sign-image and detector weights whose label map contains "
            "--source-class."
            if key == "speed_limit"
            else ""
        )
        raise FileNotFoundError(f"Sign image not found: {day_path}.{profile_hint}")

    active_path = Path(default_active)
    if not active_path.is_file():
        active_path = day_path

    resolved_source = source_class if source_class is not None else default_source
    return SignAssets(
        profile=key,
        day_image=day_path,
        active_image=active_path,
        source_class=resolved_source,
    )
