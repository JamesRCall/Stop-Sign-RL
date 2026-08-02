"""CLI for preregistered two-phase frozen-prefix risk certification.

Workflow::

    python tools/certify_attack_results.py hash-plan --plan protocol.json
    python tools/certify_attack_results.py calibrate \
        --plan protocol.json --rows calibration_rows.json \
        --out calibration_selection.json
    python tools/certify_attack_results.py certify \
        --plan protocol.json --selection calibration_selection.json \
        --rows untouched_certification_rows.json --out certificate.json

The certification rows must refer to the selection digest emitted by the
calibration command.  Existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.risk_certification import (  # noqa: E402
    calibrate_prefixes,
    canonical_sha256,
    certify_selected_prefix,
    parse_protocol,
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=_reject_constant,
                object_pairs_hook=_no_duplicate_keys,
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing certification artifact: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preregistered finite-sample sampled-population risk certification "
            "for a frozen family of traffic-sign patch prefixes."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser(
        "hash-plan",
        help="Validate a protocol and print its canonical SHA-256.",
    )
    hash_parser.add_argument("--plan", required=True)

    calibration_parser = subparsers.add_parser(
        "calibrate",
        help=(
            "Evaluate every frozen prefix on the preregistered calibration rows "
            "and seal the smallest passing prefix."
        ),
    )
    calibration_parser.add_argument("--plan", required=True)
    calibration_parser.add_argument("--rows", required=True)
    calibration_parser.add_argument("--out", required=True)

    certification_parser = subparsers.add_parser(
        "certify",
        help=(
            "Evaluate only the sealed prefix on the separate preregistered "
            "certification rows."
        ),
    )
    certification_parser.add_argument("--plan", required=True)
    certification_parser.add_argument("--selection", required=True)
    certification_parser.add_argument("--rows", required=True)
    certification_parser.add_argument("--out", required=True)
    return parser


def _attach_file_hashes(
    report: Mapping[str, Any],
    *,
    plan_path: Path,
    rows_path: Path,
    selection_path: Optional[Path] = None,
) -> dict:
    enriched = dict(report)
    files = {
        "protocol_path": str(plan_path.resolve()),
        "protocol_file_sha256": _raw_sha256(plan_path),
        "results_path": str(rows_path.resolve()),
        "results_file_sha256": _raw_sha256(rows_path),
    }
    if selection_path is not None:
        files.update(
            {
                "selection_path": str(selection_path.resolve()),
                "selection_file_sha256": _raw_sha256(selection_path),
            }
        )
    enriched["input_files"] = files
    # The core report's integrity hash covers semantic content.  File hashes are
    # an outer transport record and intentionally do not rewrite that digest.
    enriched["transport_envelope_sha256"] = canonical_sha256(enriched)
    return enriched


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan)
    plan_payload = _read_json(plan_path, label="protocol")
    protocol = parse_protocol(plan_payload)

    if args.command == "hash-plan":
        print(protocol.canonical_sha256)
        return 0

    rows_path = Path(args.rows)
    rows_payload = _read_json(rows_path, label=f"{args.command} rows")
    out_path = Path(args.out)
    if args.command == "calibrate":
        report = calibrate_prefixes(plan_payload, rows_payload)
        report = _attach_file_hashes(
            report,
            plan_path=plan_path,
            rows_path=rows_path,
        )
        _write_new_json(out_path, report)
        selection = report["selection"]
        print(
            "[CALIBRATION] "
            f"status={selection['status']} "
            f"prefix={selection['selected_prefix_id']} "
            f"selection_sha256={report['selection_sha256']}"
        )
        print(f"[CALIBRATION] wrote {out_path}")
        return 0

    selection_path = Path(args.selection)
    selection_payload = _read_json(selection_path, label="calibration selection")
    # A CLI report carries an outer transport envelope not present in the core
    # report hash.  Remove only these two known transport fields before strict
    # core verification; no other selection fields are altered.
    core_selection = dict(selection_payload)
    declared_transport_hash = core_selection.pop("transport_envelope_sha256", None)
    if declared_transport_hash is not None:
        if not isinstance(declared_transport_hash, str):
            raise ValueError("selection transport_envelope_sha256 must be a string")
        if declared_transport_hash != canonical_sha256(core_selection):
            raise ValueError("selection transport envelope integrity hash is invalid")
    core_selection.pop("input_files", None)
    report = certify_selected_prefix(plan_payload, core_selection, rows_payload)
    report = _attach_file_hashes(
        report,
        plan_path=plan_path,
        rows_path=rows_path,
        selection_path=selection_path,
    )
    _write_new_json(out_path, report)
    print(
        "[CERTIFICATION] "
        f"status={report['certificate_status']} "
        f"prefix={report['selection']['selected_prefix_id']} "
        f"worst_joint_lower="
        f"{report['aggregation']['primary_worst_joint_lower_bound']:.6f}"
    )
    print(f"[CERTIFICATION] wrote {out_path}")
    return 0 if bool(report["certificate_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
