"""Reproducibility manifest helpers for paper-facing experiments."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Union


def sha256_file(path: Union[str, Path]) -> Optional[str]:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Union[str, Path]) -> Dict[str, Any]:
    file_path = Path(path).resolve()
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else None,
        "sha256": sha256_file(file_path),
    }


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_experiment_manifest(
    *,
    config: Mapping[str, Any],
    sign_day_path: Union[str, Path],
    sign_active_path: Union[str, Path],
    pole_path: Optional[Union[str, Path]],
    background_paths: Iterable[Union[str, Path]],
    weights_path: Optional[Union[str, Path]],
    source_class: Any,
    attack_target_class: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv),
        "config": dict(config),
        "resolved": {
            "source_class": str(source_class),
            "attack_target_class": (
                str(attack_target_class) if attack_target_class is not None else None
            ),
            "sign_day": file_record(sign_day_path),
            "sign_active": file_record(sign_active_path),
            "pole": file_record(pole_path) if pole_path else None,
            "backgrounds": [file_record(path) for path in background_paths],
            "detector_weights": file_record(weights_path) if weights_path else None,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "numpy",
                    "Pillow",
                    "torch",
                    "torchvision",
                    "gymnasium",
                    "stable-baselines3",
                    "sb3-contrib",
                    "ultralytics",
                    "transformers",
                ]
            ),
        },
    }


def write_manifest(path: Union[str, Path], manifest: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
    os.replace(temp, output)
