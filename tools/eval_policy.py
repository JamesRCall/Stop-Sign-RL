"""Quick evaluation script for the stop-sign grid PPO policy.

Loads the newest checkpoint (or a specified model), runs deterministic eval
episodes, and logs scalars (and optional images) to TensorBoard.
If VecNormalize was used in training, pass or auto-detect the saved stats.
"""
import os
import sys
import argparse
import json
import re
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecNormalize

from envs.stop_sign_grid_env import StopSignGridEnv
from train_single_stop_sign import (
    build_backgrounds,
    resolve_paint_list,
    resolve_yolo_weights,
    find_latest_checkpoint,
)


def make_env(
    args,
    stop_plain: Image.Image,
    stop_uv: Image.Image,
    pole_rgba: Optional[Image.Image],
    img_size: Tuple[int, int],
    use_vecnorm: bool,
):
    backgrounds = build_backgrounds(args.bg_mode, args.bgdir, img_size)
    paint_list = resolve_paint_list(args.paint, args.paint_list)
    env = StopSignGridEnv(
        stop_sign_image=stop_plain,
        stop_sign_uv_image=stop_uv,
        background_images=backgrounds,
        pole_image=pole_rgba,
        yolo_weights=args.yolo_weights,
        yolo_device=args.detector_device,
        detector_type=str(args.detector),
        detector_model=str(args.detector_model) if args.detector_model else None,
        img_size=img_size,
        obs_size=(int(args.obs_size), int(args.obs_size)),
        obs_margin=float(args.obs_margin),
        obs_include_mask=bool(int(args.obs_include_mask)),
        steps_per_episode=int(args.episode_steps),
        eval_K=int(args.eval_K),
        detector_debug=False,
        grid_cell_px=int(args.grid_cell),
        uv_paint=paint_list[0],
        uv_paint_list=paint_list if len(paint_list) > 1 else None,
        use_single_color=True,
        cell_cover_thresh=float(args.cell_cover_thresh),
        uv_drop_threshold=float(args.uv_threshold),
        success_conf_threshold=float(args.success_conf),
        lambda_efficiency=float(args.lambda_efficiency),
        efficiency_eps=float(args.efficiency_eps),
        transform_strength=float(args.transform_strength),
        fixed_angle_deg=(float(args.fixed_angle_deg) if args.fixed_angle_deg is not None else None),
        day_tolerance=0.05,
        lambda_day=float(args.lambda_day),
        lambda_area=float(args.lambda_area),
        area_target_frac=float(args.area_target) if args.area_target is not None else None,
        step_cost=float(args.step_cost),
        step_cost_after_target=float(args.step_cost_after_target),
        lambda_iou=float(args.lambda_iou),
        lambda_misclass=float(args.lambda_misclass),
        lambda_perceptual=float(args.lambda_perceptual),
        area_cap_frac=float(args.area_cap_frac) if args.area_cap_frac and float(args.area_cap_frac) > 0 else None,
        area_cap_penalty=float(args.area_cap_penalty),
        area_cap_mode=str(args.area_cap_mode),
    )
    env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
    v = DummyVecEnv([lambda: env])
    v = VecTransposeImage(v)
    if use_vecnorm:
        if args.vecnorm:
            v = VecNormalize.load(args.vecnorm, v)
        else:
            v = VecNormalize(v, norm_obs=True, norm_reward=False, clip_obs=5.0)
        v.training = False
        v.norm_reward = False
    return v


def parse_args():
    ap = argparse.ArgumentParser("Evaluate stop-sign PPO policy")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    ap.add_argument("--no-pole", action="store_true")
    ap.add_argument("--yolo-version", choices=["8", "11"], default="8")
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--detector-device", default=os.getenv("YOLO_DEVICE", "auto"))
    ap.add_argument("--detector", default="yolo",
                    help="Detector backend: yolo, torchvision, or rtdetr.")
    ap.add_argument("--detector-model", default="",
                    help="Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).")
    ap.add_argument("--ckpt", default="./_runs/checkpoints")
    ap.add_argument("--model", default=None, help="Path to model .zip (defaults to latest in --ckpt)")
    ap.add_argument("--vecnorm", default=None, help="Path to VecNormalize stats .pkl (optional)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None,
                    help="Base seed for eval episodes (episode i uses seed+ i).")
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--tb", default="./_runs/tb_eval", help="TensorBoard log dir (optional).")
    ap.add_argument("--tb-tag", default="eval", help="TensorBoard tag prefix.")
    ap.add_argument("--log-images", type=int, default=10, help="Max eval images to log to TensorBoard.")
    ap.add_argument("--out-json", default="", help="Optional path to write eval summary JSON.")
    ap.add_argument("--out-episodes-json", default="", help="Optional path to write per-episode rows JSON.")
    ap.add_argument("--save-overlay-dir", default="",
                    help="Optional directory to save per-episode overlay pattern PNGs (info['overlay_pil']).")
    ap.add_argument("--save-composited-dir", default="",
                    help="Optional directory to save per-episode composited preview PNGs (info['composited_pil']).")

    # Env settings (match training defaults)
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--eval-K", type=int, default=3)
    ap.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    ap.add_argument("--uv-threshold", type=float, default=0.75)
    ap.add_argument("--success-conf", type=float, default=0.20)
    ap.add_argument("--lambda-area", type=float, default=0.70)
    ap.add_argument("--lambda-iou", type=float, default=0.40)
    ap.add_argument("--lambda-misclass", type=float, default=0.60)
    ap.add_argument("--lambda-perceptual", type=float, default=0.0)
    ap.add_argument("--lambda-day", type=float, default=0.0)
    ap.add_argument("--lambda-efficiency", type=float, default=0.40)
    ap.add_argument("--efficiency-eps", type=float, default=0.02)
    ap.add_argument("--area-target", type=float, default=0.25)
    ap.add_argument("--step-cost", type=float, default=0.012)
    ap.add_argument("--step-cost-after-target", type=float, default=0.14)
    ap.add_argument("--transform-strength", type=float, default=1.0)
    ap.add_argument("--fixed-angle-deg", type=float, default=None,
                    help="If set, use a fixed sign rotation angle (degrees) for all transforms.")
    ap.add_argument("--area-cap-frac", type=float, default=0.30)
    ap.add_argument("--area-cap-penalty", type=float, default=-0.20)
    ap.add_argument("--area-cap-mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--obs-size", type=int, default=224)
    ap.add_argument("--obs-margin", type=float, default=0.10)
    ap.add_argument("--obs-include-mask", type=int, default=1)
    ap.add_argument("--paint", default="yellow")
    ap.add_argument("--paint-list", default="")
    ap.add_argument("--cell-cover-thresh", type=float, default=0.60)
    return ap.parse_args()


def _sanitize_tb_component(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    name = name.strip("._-")
    return name if name else "eval"


def _finite_or_nan(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _mean(vals: List[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals: List[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.std(vals)) if vals else float("nan")


def _median(vals: List[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.median(vals)) if vals else float("nan")


def _safe_int_list(value: Any) -> List[int]:
    if not isinstance(value, (list, tuple)):
        return []
    out: List[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def main():
    args = parse_args()
    args.yolo_weights = resolve_yolo_weights(args.yolo_version, args.yolo_weights)
    if args.model is None:
        args.model = find_latest_checkpoint(args.ckpt)
    if not args.model:
        raise FileNotFoundError("No model checkpoint found; pass --model or place .zip in --ckpt.")

    stop_plain = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    stop_uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(stop_uv_path).convert("RGBA") if os.path.exists(stop_uv_path) else stop_plain.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole_rgba = Image.open(pole_path).convert("RGBA") if (not args.no_pole and os.path.exists(pole_path)) else None
    img_size = (640, 640)

    model = MaskablePPO.load(args.model, env=None, device="auto")
    obs_space = model.observation_space
    use_vecnorm = hasattr(obs_space, "low") and float(np.min(obs_space.low)) < 0.0
    if use_vecnorm and not args.vecnorm:
        print("[EVAL] WARN: model expects normalized obs but no vecnorm stats provided; using fresh VecNormalize.")

    env = make_env(args, stop_plain, stop_uv, pole_rgba, img_size, use_vecnorm=use_vecnorm)
    model.set_env(env)

    writer = None
    tb_dir = ""
    if args.tb:
        tb_root = os.path.abspath(args.tb)
        os.makedirs(tb_root, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{_sanitize_tb_component(args.tb_tag)}_{stamp}"
        tb_dir = os.path.join(tb_root, run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"[EVAL] tb_run_dir={tb_dir}")

    overlay_save_dir = os.path.abspath(args.save_overlay_dir) if args.save_overlay_dir else ""
    composited_save_dir = os.path.abspath(args.save_composited_dir) if args.save_composited_dir else ""
    if overlay_save_dir:
        os.makedirs(overlay_save_dir, exist_ok=True)
        print(f"[EVAL] save_overlay_dir={overlay_save_dir}")
    if composited_save_dir:
        os.makedirs(composited_save_dir, exist_ok=True)
        print(f"[EVAL] save_composited_dir={composited_save_dir}")

    image_budget = int(max(0, args.log_images))
    successes = 0
    steps_list: List[float] = []
    base_list: List[float] = []
    area_list: List[float] = []
    after_list: List[float] = []
    drop_on_list: List[float] = []
    iou_list: List[float] = []
    misclass_list: List[float] = []
    efficiency_list: List[float] = []
    selected_cells_list: List[float] = []
    episode_runtime_sec_list: List[float] = []
    episode_rows: List[Dict[str, Any]] = []
    eval_t0 = time.perf_counter()

    for ep_idx in range(int(args.episodes)):
        ep_t0 = time.perf_counter()
        seed_i = None
        if args.seed is not None:
            seed_i = int(args.seed) + int(ep_idx)
            # SB3 VecNormalize may not accept seed in reset; try env_method fallback.
            try:
                obs = env.reset(seed=[seed_i])
            except TypeError:
                try:
                    obs = env.reset(seed=seed_i)
                except TypeError:
                    env.env_method("reset", seed=seed_i)
                    obs = env.reset()
        else:
            obs = env.reset()
        done = False
        steps = 0
        last_info = {}
        while not done:
            masks = env.env_method("action_masks")
            action, _ = model.predict(obs, deterministic=bool(int(args.deterministic)), action_masks=masks[0])
            obs, reward, done, info = env.step(action)
            steps += 1
            last_info = info[0] if isinstance(info, list) and info else info

        info_d = last_info if isinstance(last_info, dict) else {}
        trace_d = info_d.get("trace", {})
        trace_d = trace_d if isinstance(trace_d, dict) else {}
        trace_selected_indices = _safe_int_list(trace_d.get("selected_indices", []))
        trace_transform_seeds = _safe_int_list(trace_d.get("transform_seeds", []))
        trace_place_seed = trace_d.get("place_seed", None)
        try:
            trace_place_seed = int(trace_place_seed) if trace_place_seed is not None else None
        except (TypeError, ValueError):
            trace_place_seed = None
        success = bool(info_d.get("uv_success", False))
        base_conf = _finite_or_nan(info_d.get("base_conf", info_d.get("c0_day", np.nan)))
        after_conf = _finite_or_nan(info_d.get("after_conf", info_d.get("c_on", np.nan)))
        drop_on = _finite_or_nan(info_d.get("drop_on", np.nan))
        drop_day = _finite_or_nan(info_d.get("drop_day", np.nan))
        area_frac = _finite_or_nan(info_d.get("total_area_mask_frac", np.nan))
        mean_iou = _finite_or_nan(info_d.get("mean_iou", np.nan))
        misclass_rate = _finite_or_nan(info_d.get("misclass_rate", np.nan))
        selected_cells = _finite_or_nan(info_d.get("selected_cells", np.nan))
        reward_final = _finite_or_nan(info_d.get("reward", np.nan))
        eval_k_used = _finite_or_nan(info_d.get("eval_K_used", np.nan))
        ep_runtime_sec = float(time.perf_counter() - ep_t0)
        drop_per_area = float("nan")
        if not np.isnan(drop_on) and not np.isnan(area_frac) and area_frac > 0:
            drop_per_area = float(drop_on / area_frac)

        successes += 1 if success else 0
        steps_list.append(float(steps))
        base_list.append(base_conf)
        area_list.append(area_frac)
        after_list.append(after_conf)
        drop_on_list.append(drop_on)
        iou_list.append(mean_iou)
        misclass_list.append(misclass_rate)
        efficiency_list.append(drop_per_area)
        selected_cells_list.append(selected_cells)
        episode_runtime_sec_list.append(ep_runtime_sec)

        overlay_img_path = ""
        composited_img_path = ""
        seed_suffix = f"_seed_{int(seed_i)}" if seed_i is not None else ""
        if overlay_save_dir:
            overlay_img = info_d.get("overlay_pil", None)
            if overlay_img is not None and hasattr(overlay_img, "save"):
                overlay_img_path = os.path.join(overlay_save_dir, f"ep_{int(ep_idx):04d}{seed_suffix}_overlay.png")
                overlay_img.save(overlay_img_path, format="PNG")
        if composited_save_dir:
            composited_img = info_d.get("composited_pil", None)
            if composited_img is not None and hasattr(composited_img, "save"):
                composited_img_path = os.path.join(composited_save_dir, f"ep_{int(ep_idx):04d}{seed_suffix}_composited.png")
                composited_img.save(composited_img_path, format="PNG")

        episode_rows.append({
            "episode_index": int(ep_idx),
            "seed": int(seed_i) if seed_i is not None else None,
            "success": bool(success),
            "steps": int(steps),
            "base_conf": base_conf,
            "after_conf": after_conf,
            "drop_on": drop_on,
            "drop_day": drop_day,
            "area_frac": area_frac,
            "drop_per_area": drop_per_area,
            "mean_iou": mean_iou,
            "misclass_rate": misclass_rate,
            "selected_cells": selected_cells,
            "reward_final": reward_final,
            "eval_K_used": eval_k_used,
            "runtime_sec": ep_runtime_sec,
            "runtime_per_step_sec": float(ep_runtime_sec / steps) if steps > 0 else float("nan"),
            "area_cap_exceeded": bool(info_d.get("area_cap_exceeded", False)),
            "note": str(info_d.get("note")) if "note" in info_d else "",
            "trace": {
                "phase": str(trace_d.get("phase", "")) if trace_d else "",
                "grid_cell_px": int(trace_d.get("grid_cell_px")) if trace_d.get("grid_cell_px") is not None else None,
                "selected_indices": trace_selected_indices,
                "place_seed": trace_place_seed,
                "transform_seeds": trace_transform_seeds,
            },
            "overlay_image_path": overlay_img_path,
            "composited_image_path": composited_img_path,
        })

        if writer is not None:
            tag = str(args.tb_tag)
            writer.add_scalar(f"{tag}/episode_steps", steps, ep_idx)
            if not np.isnan(area_frac):
                writer.add_scalar(f"{tag}/episode_area_frac", area_frac, ep_idx)
            if not np.isnan(after_conf):
                writer.add_scalar(f"{tag}/episode_after_conf", after_conf, ep_idx)
            if not np.isnan(drop_on):
                writer.add_scalar(f"{tag}/episode_drop_on", drop_on, ep_idx)
            if not np.isnan(mean_iou):
                writer.add_scalar(f"{tag}/episode_mean_iou", mean_iou, ep_idx)
            if not np.isnan(misclass_rate):
                writer.add_scalar(f"{tag}/episode_misclass_rate", misclass_rate, ep_idx)
            if not np.isnan(drop_per_area):
                writer.add_scalar(f"{tag}/episode_drop_per_area", drop_per_area, ep_idx)
            writer.add_scalar(f"{tag}/episode_runtime_sec", ep_runtime_sec, ep_idx)
            writer.add_scalar(f"{tag}/episode_success", 1.0 if success else 0.0, ep_idx)
            if image_budget > 0:
                img = info_d.get("composited_pil", None)
                if img is not None:
                    from utils.tb_callbacks import pil_to_chw_uint8
                    chw = pil_to_chw_uint8(img)
                    writer.add_image(f"{tag}/eval_overlay_img", chw, ep_idx, dataformats="CHW")
                    image_budget -= 1
            writer.flush()

    success_rate = successes / float(args.episodes)
    mean_steps = _mean(steps_list)
    mean_base = _mean(base_list)
    mean_area = _mean(area_list)
    mean_after = _mean(after_list)
    mean_drop_on = _mean(drop_on_list)
    mean_iou = _mean(iou_list)
    mean_misclass = _mean(misclass_list)
    mean_drop_per_area = _mean(efficiency_list)
    mean_selected_cells = _mean(selected_cells_list)
    mean_runtime_sec = _mean(episode_runtime_sec_list)
    total_runtime_sec = float(time.perf_counter() - eval_t0)
    total_steps = float(np.nansum(steps_list)) if steps_list else 0.0
    runtime_per_step_sec = float(total_runtime_sec / total_steps) if total_steps > 0 else float("nan")

    print(f"[EVAL] model={args.model}")
    print(f"[EVAL] episodes={args.episodes} success_rate={success_rate:.3f}")
    print(f"[EVAL] mean_steps={mean_steps:.2f} mean_area_frac={mean_area:.4f} mean_after_conf={mean_after:.4f}")
    print(f"[EVAL] mean_drop_on={mean_drop_on:.4f} mean_drop_per_area={mean_drop_per_area:.4f}")
    print(f"[EVAL] mean_iou={mean_iou:.4f} mean_misclass_rate={mean_misclass:.4f}")
    print(f"[EVAL] runtime_total_sec={total_runtime_sec:.2f} runtime_per_episode_sec={mean_runtime_sec:.3f}")

    if writer is not None:
        tag = str(args.tb_tag)
        writer.add_scalar(f"{tag}/success_rate", success_rate, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_steps", mean_steps, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_area_frac", mean_area, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_after_conf", mean_after, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_drop_on", mean_drop_on, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_drop_per_area", mean_drop_per_area, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_iou", mean_iou, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_misclass_rate", mean_misclass, int(args.episodes))
        writer.add_scalar(f"{tag}/runtime_total_sec", total_runtime_sec, int(args.episodes))
        writer.add_scalar(f"{tag}/runtime_per_episode_sec", mean_runtime_sec, int(args.episodes))
        writer.add_scalar(f"{tag}/runtime_per_step_sec", runtime_per_step_sec, int(args.episodes))
        writer.flush()
        writer.close()

    out = {
        "method": "ppo",
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": args.model,
        "model_abs": os.path.abspath(args.model),
        "vecnorm": args.vecnorm,
        "vecnorm_abs": os.path.abspath(args.vecnorm) if args.vecnorm else "",
        "tb_run_dir": tb_dir,
        "save_overlay_dir": overlay_save_dir,
        "save_composited_dir": composited_save_dir,
        "episodes": int(args.episodes),
        "n_success": int(successes),
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "std_steps": _std(steps_list),
        "median_steps": _median(steps_list),
        "mean_base_conf": mean_base,
        "std_base_conf": _std(base_list),
        "mean_area_frac": mean_area,
        "std_area_frac": _std(area_list),
        "mean_after_conf": mean_after,
        "std_after_conf": _std(after_list),
        "mean_drop_on": mean_drop_on,
        "std_drop_on": _std(drop_on_list),
        "mean_drop_per_area": mean_drop_per_area,
        "std_drop_per_area": _std(efficiency_list),
        "mean_iou": mean_iou,
        "std_iou": _std(iou_list),
        "mean_misclass_rate": mean_misclass,
        "std_misclass_rate": _std(misclass_list),
        "mean_selected_cells": mean_selected_cells,
        "std_selected_cells": _std(selected_cells_list),
        "runtime_total_sec": total_runtime_sec,
        "runtime_per_episode_mean_sec": mean_runtime_sec,
        "runtime_per_episode_std_sec": _std(episode_runtime_sec_list),
        "runtime_per_step_sec": runtime_per_step_sec,
        "episodes_detail": episode_rows,
        "argv": list(sys.argv),
        "config": vars(args),
    }

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"[EVAL] wrote summary json: {args.out_json}")

    if args.out_episodes_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_episodes_json)), exist_ok=True)
        with open(args.out_episodes_json, "w", encoding="utf-8") as f:
            json.dump(episode_rows, f, indent=2)
        print(f"[EVAL] wrote episode json: {args.out_episodes_json}")


if __name__ == "__main__":
    main()
