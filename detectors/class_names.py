"""Utilities for resolving detector class references without silent fallbacks."""
from __future__ import annotations

from difflib import get_close_matches
from numbers import Integral
import re
from typing import Mapping, Union


ClassReference = Union[str, int]


def normalize_class_name(value: str) -> str:
    """Normalize spacing and punctuation while preserving alphanumeric content."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def resolve_class_id(
    id_to_name: Mapping[int, str],
    class_ref: ClassReference,
    *,
    role: str = "class",
) -> int:
    """Resolve an explicit class id or a normalized exact class name.

    A missing name is an experiment-configuration error.  Silently substituting a
    COCO id can make a run appear successful while optimizing the wrong class, so
    this helper deliberately fails with an actionable message instead.
    """
    names_by_id = {int(k): str(v) for k, v in dict(id_to_name or {}).items()}

    def validate_numeric(class_id: int) -> int:
        if names_by_id and int(class_id) not in names_by_id:
            preview = ", ".join(
                f"{idx}:{name}" for idx, name in sorted(names_by_id.items())[:20]
            )
            raise ValueError(
                f"Could not resolve {role} id {class_id}; it is absent from the "
                f"detector label map. Available ids: {preview}."
            )
        return int(class_id)

    if isinstance(class_ref, Integral) and not isinstance(class_ref, bool):
        return validate_numeric(int(class_ref))

    raw = str(class_ref).strip()
    if not raw:
        raise ValueError(f"{role} must be a non-empty class name or integer id")
    if re.fullmatch(r"[+-]?\d+", raw):
        return validate_numeric(int(raw))

    wanted = normalize_class_name(raw)
    normalized = {
        normalize_class_name(name): int(class_id)
        for class_id, name in names_by_id.items()
    }
    if wanted in normalized:
        return normalized[wanted]

    names = [str(name) for _, name in sorted(names_by_id.items())]
    suggestions = get_close_matches(raw.lower(), [n.lower() for n in names], n=3, cutoff=0.45)
    hint = f" Closest labels: {', '.join(suggestions)}." if suggestions else ""
    preview = ", ".join(names[:20])
    if len(names) > 20:
        preview += ", ..."
    available = f" Available labels: {preview}." if preview else " The detector exposed no label map."
    raise ValueError(
        f"Could not resolve {role} {raw!r}.{hint}{available} "
        "Pass an integer class id only when the detector label map is unavailable."
    )
