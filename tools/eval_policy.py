"""Quick evaluation script for the stop-sign grid PPO policy."""
import os
import sys
import argparse
from typing import Optional, Tuple

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
    ap.add_argument("--ckpt", default="./_runs/checkpoints")
    ap.add_argument("--model", default=None, help="Path to model .zip (defaults to latest in --ckpt)")
    ap.add_argument("--vecnorm", default=None, help="Path to VecNormalize stats .pkl (optional)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--tb", default="./_runs/tb_eval", help="TensorBoard log dir (optional).")
    ap.add_argument("--tb-tag", default="eval", help="TensorBoard tag prefix.")
    ap.add_argument("--log-images", type=int, default=10, help="Max eval images to log to TensorBoard.")

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
    if args.tb:
        tb_dir = os.path.abspath(args.tb)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    image_budget = int(max(0, args.log_images))
    successes = 0
    steps_list = []
    area_list = []
    after_list = []

    for ep_idx in range(int(args.episodes)):
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

        steps_list.append(steps)
        if isinstance(last_info, dict):
            successes += 1 if last_info.get("uv_success", False) else 0
            area_list.append(float(last_info.get("total_area_mask_frac", np.nan)))
            after_list.append(float(last_info.get("after_conf", last_info.get("c_on", np.nan))))
            if writer is not None:
                tag = str(args.tb_tag)
                writer.add_scalar(f"{tag}/episode_steps", steps, ep_idx)
                writer.add_scalar(f"{tag}/episode_area_frac", area_list[-1], ep_idx)
                writer.add_scalar(f"{tag}/episode_after_conf", after_list[-1], ep_idx)
                writer.add_scalar(
                    f"{tag}/episode_success",
                    1.0 if last_info.get("uv_success", False) else 0.0,
                    ep_idx,
                )
                if image_budget > 0 and isinstance(last_info, dict):
                    img = last_info.get("composited_pil", None)
                    if img is not None:
                        from utils.tb_callbacks import pil_to_chw_uint8
                        chw = pil_to_chw_uint8(img)
                        writer.add_image(f"{tag}/eval_overlay_img", chw, ep_idx, dataformats="CHW")
                        image_budget -= 1
                writer.flush()

    def _mean(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    success_rate = successes / float(args.episodes)
    mean_steps = _mean(steps_list)
    mean_area = _mean(area_list)
    mean_after = _mean(after_list)

    print(f"[EVAL] model={args.model}")
    print(f"[EVAL] episodes={args.episodes} success_rate={success_rate:.3f}")
    print(f"[EVAL] mean_steps={mean_steps:.2f} mean_area_frac={mean_area:.4f} mean_after_conf={mean_after:.4f}")

    if writer is not None:
        tag = str(args.tb_tag)
        writer.add_scalar(f"{tag}/success_rate", success_rate, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_steps", mean_steps, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_area_frac", mean_area, int(args.episodes))
        writer.add_scalar(f"{tag}/mean_after_conf", mean_after, int(args.episodes))
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
