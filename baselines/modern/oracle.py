"""Measured, deadline-aware detector oracle shared by the two new attacks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
from PIL import Image


class SearchStopped(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class RecordingDetector:
    def __init__(self, detector):
        self.detector = detector
        self.records = []
        self.images = 0

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def infer_detections_batch(self, images):
        output = self.detector.infer_detections_batch(images)
        if len(output) != len(images):
            raise RuntimeError("Detector returned the wrong number of results")
        self.images += len(images)
        self.records.extend(output)
        self.records = self.records[-64:]
        return output


def evaluate_metrics(c0_day, c0_on, c_day, c_on, area, config):
    """Paper confidence criterion, plus separately reported material/day checks.

    Keep the legacy soft 0.30 penalty; imposing a new hard 0.30 area limit on
    only the modern attacks would bias comparison with the stored old results.
    """
    drop_day = float(c0_day - c_day)
    drop_on = float(c0_day - c_on)  # Legacy paper reward uses the daylight reference.
    threshold = float(config.get("success_conf", 0.20))
    eligible = bool(c0_day > threshold and c0_on > threshold)
    success = bool(eligible and c_on <= threshold)
    target = float(config.get("area_target", 0.25))
    weight = float(config.get("lambda_area", 0.7))
    raw = min(drop_on, max(0.0, c0_day - threshold)) - weight * area
    raw -= float(config.get("lambda_day", 0.0)) * max(0.0, drop_day - float(config.get("day_tolerance", 0.05)))
    excess = max(0.0, area - target)
    raw -= weight * (4.5 * excess + excess**2)
    raw += float(config.get("lambda_efficiency", 0.4)) * np.log1p(max(0.0, drop_on) / max(area, float(config.get("efficiency_eps", 0.02))))
    # All optimizers receive this same score. Localization/misclassification
    # reward terms are deliberately absent: score-only detector adaptation.
    cap = float(config.get("area_cap_frac", 0.30) or 1.0)
    over = bool(area > cap)
    if over and config.get("area_cap_mode", "soft") == "soft":
        raw = -abs(float(config.get("area_cap_penalty", -0.2))) * (1 + 2 * (area - cap) / cap)
    objective = config.get("search_objective", "dirp_score")
    if objective not in ("dirp_score", "suppression"):
        raise ValueError("Unknown search objective")
    # Confidence-only search is a diagnostic alternative, not the old reward.
    # Keep all original metrics and gates; never redefine attack success.
    optimization_score = -c_on if objective == "suppression" else raw
    return dict(score=float(raw), optimization_score=float(optimization_score),
                base_conf=float(c0_day), base_uv_conf=float(c0_on),
                after_conf=float(c_on), patched_day_conf=float(c_day), drop_day=drop_day,
                drop_on=float(c0_on - c_on), area_frac=float(area), success=success,
                clean_eligible=eligible, area_cap_exceeded=over,
                day_preserved=bool(drop_day <= float(config.get("day_tolerance", 0.05))),
                joint_success=bool(success and not over and drop_day <= float(config.get("day_tolerance", 0.05))))


class AttackOracle:
    def __init__(self, env, recorder, *, seed, config, candidate_limit, deadline,
                 folder, progress=None):
        self.env, self.recorder = env, recorder
        self.config, self.candidate_limit = config, int(candidate_limit)
        self.deadline, self.folder, self.progress = deadline, Path(folder), progress
        self.folder.mkdir(parents=True, exist_ok=True)
        self.shape = (env.sign_rgba_day.height, env.sign_rgba_day.width)
        self.sign_mask = np.asarray(env._sign_alpha) > 0
        self.sign_pixels = int(self.sign_mask.sum())
        self.grid_cell = int(env.grid_cell_px)
        self.area_target = float(config.get("area_target", 0.25))
        self.evaluations = 0
        self.best = None
        self.best_key = None
        self.min_confidence = 1.0
        self.zero_area_candidates = 0
        self.search_stage = "unspecified"
        self.started = time.monotonic()
        self.trace_path = self.folder / "trace.jsonl"
        self.original_overlay = env._apply_grid_overlay
        self.current_mask = np.zeros(self.shape, dtype=bool)
        self.current_rgb = np.zeros((*self.shape, 3), dtype=np.uint8)
        self.queries_at_start = recorder.images
        env.reset(seed=int(seed))
        self.material_rgb = np.asarray(env.paint.active_rgb, dtype=np.float32)
        initial = recorder.records[-2 * env.eval_K_max:]
        if len(initial) != 2 * env.eval_K_max:
            raise RuntimeError("Unexpected clean-reference query count")
        self.k = int(env.eval_K_max)
        self.c0_day = float(np.mean([d["target_conf"] for d in initial[:self.k]]))
        self.c0_on = float(np.mean([d["target_conf"] for d in initial[self.k:]]))
        self.initial_queries = recorder.images - self.queries_at_start
        if self.initial_queries != 2 * self.k:
            raise RuntimeError("Reference acquisition must cost exactly 2*K images")

        def overlay(bound_env, sign_rgba, mode):
            rgb = np.array(sign_rgba.convert("RGB"), copy=True)
            paint_color, alpha = bound_env._paint_color_and_alpha(bound_env.paint, mode)
            alpha_mask = self.current_mask.astype(np.float32) * (np.asarray(bound_env._sign_alpha, dtype=np.float32)/255) * alpha
            # Match PIL's byte alpha quantization in the original renderer.
            mask = Image.fromarray((alpha_mask * 255).astype(np.uint8))
            result = Image.fromarray(rgb)
            color = Image.fromarray(self.current_rgb) if mode == "on" else tuple(paint_color)
            result.paste(color, mask=mask)
            return Image.merge("RGBA", (*result.split(), sign_rgba.getchannel("A")))

        env._apply_grid_overlay = types.MethodType(overlay, env)
        self.contract = dict(scene_seed=int(seed), place_seed=int(env._place_seed),
                             transform_seeds=list(map(int, env._transform_seeds)),
                             image_size=list(env.img_size), sign_size=list(env.sign_rgba_day.size),
                             sign_day_sha256=hashlib.sha256(env.sign_rgba_day.tobytes()).hexdigest(),
                             sign_uv_sha256=hashlib.sha256(env.sign_rgba_on.tobytes()).hexdigest(),
                             background_sha256=hashlib.sha256(env._bg_rgb.tobytes()).hexdigest(),
                             grid_cell_pixels=self.grid_cell, candidate_limit=self.candidate_limit,
                             query_limit=2*self.k*(self.candidate_limit+1),
                             area_cap_mode=config.get("area_cap_mode", "soft"),
                             area_cap_fraction=config.get("area_cap_frac", 0.3),
                             adapter_revision=4, circle_centers="repaired onto opaque sign support",
                             impact_cell_rule="actual cropped-cell coverage, identical to DIRP",
                             valid_grid_cell_count=int(env._n_valid),
                             cell_cover_threshold=float(env.cell_cover_thresh),
                             search_objective=config.get("search_objective", "dirp_score"),
                             color_mode=config.get("color_mode", "rgb"),
                             search_area_levels=config.get("search_area_levels", [.25, .5]),
                             active_rgb_optimized=config.get("color_mode", "rgb") == "rgb",
                             material_active_rgb=list(map(float, self.material_rgb)),
                             inactive_color=list(env.paint.day_rgb))
        atomic_json(self.folder / "contract.json", self.contract)

    @property
    def queries(self):
        return self.recorder.images - self.queries_at_start

    def evaluate(self, mask, rgb):
        if time.monotonic() >= self.deadline:
            raise SearchStopped("wall_time_limit")
        if self.evaluations >= self.candidate_limit:
            raise SearchStopped("query_limit")
        mask = np.asarray(mask, dtype=bool)
        rgb = np.asarray(rgb)
        if mask.shape != self.shape or rgb.shape != (*self.shape, 3):
            raise ValueError("Candidate shape does not match sign assets")
        if not np.isfinite(rgb).all():
            raise ValueError("Non-finite candidate RGB")
        self.current_mask = mask & self.sign_mask
        self.current_rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        seeds = self.env._transform_seeds[:self.k]
        images = [self.env._render_variant(kind=kind, use_overlay=True, transform_seed=s)
                  for kind in ("day", "on") for s in seeds]
        before = self.recorder.images
        detections = self.recorder.infer_detections_batch(images)
        for image in images:
            image.close()
        if self.recorder.images - before != 2 * self.k:
            raise RuntimeError("Candidate query-accounting invariant failed")
        c_day = float(np.mean([d["target_conf"] for d in detections[:self.k]]))
        c_on = float(np.mean([d["target_conf"] for d in detections[self.k:]]))
        row = evaluate_metrics(self.c0_day, self.c0_on, c_day, c_on,
                               float(self.current_mask.sum()) / self.sign_pixels, self.config)
        self.evaluations += 1
        self.min_confidence = min(self.min_confidence, c_on)
        self.zero_area_candidates += int(row["area_frac"] == 0)
        row.update(candidate=self.evaluations, detector_image_queries=self.queries,
                   search_stage=self.search_stage,
                   elapsed_sec=time.monotonic()-self.started)
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        # Match the existing Random winner convention: a successful candidate
        # outranks a failure; among successes prefer the smaller footprint.
        key = (int(row["success"]), -row["area_frac"] if row["success"] else row["optimization_score"], -row["after_conf"])
        if self.best_key is None or key > self.best_key:
            self.best, self.best_key = dict(row), key
            np.savez_compressed(self.folder / "best_patch.npz", mask=self.current_mask, rgb=self.current_rgb)
            self._save_best_images()
            atomic_json(self.folder / "best.json", self.best)
        if self.progress is not None:
            self.progress(self)
        return row

    def _save_best_images(self):
        rgba = np.dstack((self.current_rgb, self.current_mask.astype(np.uint8)*255))
        Image.fromarray(rgba).save(self.folder / "best_overlay.png")
        for kind in ["day", "on"]:
            image = self.env._render_variant(kind=kind, use_overlay=True,
                                             transform_seed=self.env._transform_seeds[0])
            image.save(self.folder / f"best_{kind}.png")
            image.close()

    def close(self):
        self.env._apply_grid_overlay = self.original_overlay
        self.recorder.records.clear()
