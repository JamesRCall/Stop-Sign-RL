#!/usr/bin/env python3
"""Build UV-vs-Norm delta confidence JSON from real-world compare output.

Input:
  compare_real_images_detectors.py JSON containing:
    video_distance_grouped.by_condition

Output:
  Nested JSON with per-(type, daynight, detector) deltas:
    delta = uv_mean_target_conf - norm_mean_target_conf
  for overall and each distance bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _split_condition_key(key: str) -> tuple[str, str, str] | None:
    parts = str(key).split("_")
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]  # type, daynight, uvnorm


def main() -> int:
    ap = argparse.ArgumentParser(description="Create UV-minus-Norm delta confidence JSON for real-world experiments.")
    ap.add_argument(
        "--input-json",
        default="_runs/paper_data/real_detector_compare/real_world_videos_allframes_20to0.json",
        help="compare_real_images_detectors output JSON",
    )
    ap.add_argument(
        "--out-json",
        default="_runs/paper_data/real_detector_compare/real_world_delta_conf_uv_minus_norm.json",
        help="output delta JSON path",
    )
    args = ap.parse_args()

    inp = Path(args.input_json)
    if not inp.exists():
        raise FileNotFoundError(f"Input JSON not found: {inp}")

    obj = json.loads(inp.read_text(encoding="utf-8"))
    vdg = obj.get("video_distance_grouped", {}) or {}
    by_condition = vdg.get("by_condition", {}) or {}
    bins = vdg.get("distance_mapping", {}).get("bin_labels", []) or ["0-5m", "5-10m", "10-15m", "15-20m"]

    # Pair up UV/Norm conditions by (type, daynight).
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for cond_key, cond_obj in by_condition.items():
        sp = _split_condition_key(cond_key)
        if sp is None:
            continue
        typ, daynight, uvnorm = sp
        grouped.setdefault((typ, daynight), {})[uvnorm] = cond_obj

    out: dict[str, Any] = {
        "delta_definition": "uv_minus_norm",
        "input_json": str(inp),
        "bins": bins,
        "data": {},
    }

    for (typ, daynight), uvnorm_map in sorted(grouped.items()):
        uv_obj = uvnorm_map.get("uv")
        norm_obj = uvnorm_map.get("norm")
        if not uv_obj or not norm_obj:
            continue

        out["data"].setdefault(typ, {}).setdefault(daynight, {})
        uv_dets = uv_obj.get("detectors", {}) or {}
        norm_dets = norm_obj.get("detectors", {}) or {}
        det_names = sorted(set(uv_dets.keys()) | set(norm_dets.keys()))

        for det in det_names:
            uv_det = uv_dets.get(det, {}) or {}
            norm_det = norm_dets.get(det, {}) or {}

            uv_overall = uv_det.get("overall_mean_over_trials", {}) or {}
            nm_overall = norm_det.get("overall_mean_over_trials", {}) or {}

            uv_mean = _safe_float(uv_overall.get("mean_target_conf"))
            nm_mean = _safe_float(nm_overall.get("mean_target_conf"))
            overall_delta = None if (uv_mean is None or nm_mean is None) else (uv_mean - nm_mean)

            uv_bins = uv_det.get("distance_bins_mean_over_trials", {}) or {}
            nm_bins = norm_det.get("distance_bins_mean_over_trials", {}) or {}
            bin_payload: dict[str, Any] = {}
            for b in bins:
                ub = uv_bins.get(b, {}) or {}
                nb = nm_bins.get(b, {}) or {}
                u = _safe_float(ub.get("mean_target_conf"))
                n = _safe_float(nb.get("mean_target_conf"))
                d = None if (u is None or n is None) else (u - n)
                bin_payload[b] = {
                    "uv_mean_target_conf": u,
                    "norm_mean_target_conf": n,
                    "delta_uv_minus_norm": d,
                    "uv_n_trials": ub.get("n_trials"),
                    "norm_n_trials": nb.get("n_trials"),
                }

            out["data"][typ][daynight][det] = {
                "overall": {
                    "uv_mean_target_conf": uv_mean,
                    "norm_mean_target_conf": nm_mean,
                    "delta_uv_minus_norm": overall_delta,
                    "uv_n_trials": uv_overall.get("n_trials"),
                    "norm_n_trials": nm_overall.get("n_trials"),
                },
                "distance_bins": bin_payload,
            }

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SAVE] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

