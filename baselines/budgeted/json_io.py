"""Strict and crash-safe JSON helpers for comparison artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Tuple


class StrictJSONError(ValueError):
    """Raised for ambiguous or non-standard JSON input."""


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise StrictJSONError(f"non-standard JSON numeric constant: {token}")


def strict_json_loads(text: str, *, label: str = "JSON") -> Any:
    """Parse RFC-style JSON, rejecting duplicate keys and non-finite values."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJSONError(f"invalid {label}: {exc}") from exc


def strict_json_load(path: Path, *, label: str = "JSON") -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise StrictJSONError(f"cannot read {label}: {exc}") from exc
    return strict_json_loads(text, label=label)


def strict_json_dumps(value: Any) -> str:
    """Serialize standards-compliant JSON, never JavaScript NaN/Infinity."""
    try:
        return json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise StrictJSONError(f"value is not strict JSON: {exc}") from exc


def atomic_write_json_new(path: Path, value: Any) -> None:
    """Atomically publish a complete JSON file and refuse any overwrite.

    A fully flushed temporary file is hard-linked to the requested name.  Link
    creation is atomic and fails if the destination already exists.  This
    avoids both partial reports and accidental destruction of prior runs.
    """
    target = Path(path)
    payload = strict_json_dumps(value).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite existing comparison report: {target}"
            ) from exc
        except OSError as exc:
            raise OSError(
                "atomic no-overwrite publication requires same-filesystem "
                f"hard-link support: {exc}"
            ) from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
