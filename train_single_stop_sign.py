# train_single_stop_sign.py
import os
import glob
import time
import argparse
from typing import List
from PIL import Image
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback

from envs.stop_sign_env import StopSignBlobEnv
from utils.uv_paint import VIOLET_GLOW  # add more and pass them in uv_paints=[...]
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback


# ----------------- small progress logger -----------------
class ProgressETACallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every_sec: float = 10.0, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.log_every_sec = float(log_every_sec)
        self._t0 = None
        self._last_log = None

    def _on_training_start(self):
        self._t0 = time.perf_counter()
        self._last_log = self._t0

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last_log >= self.log_every_sec:
            done = self.num_timesteps
            elapsed = now - self._t0
            sps = done / max(elapsed, 1e-9)
            if self.verbose:
                eta = (self.model._total_timesteps - done) / max(sps, 1e-9)
                print(f"[{done:,}/{self.model._total_timesteps:,}] {sps:,.1f} steps/s | ETA {eta/3600:,.2f}h")
            self.logger.record("progress/steps_per_sec", sps)
            self._last_log = now
        return True


def load_backgrounds(folder: str) -> List[Image.Image]:
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs[:20]


def make_env_factory(
    stop_plain: Image.Image,
    stop_uv: Image.Image,
    pole_rgba: Image.Image,
    backgrounds: List[Image.Image],
    yolo_wts: str,
    steps_per_episode: int,
    attack_only: bool,
    attack_alpha: float,
    uv_paints=None,
    eps_day_tolerance: float = 0.03,
    day_floor: float = 0.80,
):
    uv_paints = uv_paints or [VIOLET_GLOW]

    def _init():
        return Monitor(
            StopSignBlobEnv(
                stop_sign_image=stop_plain,
                stop_sign_uv_image=stop_uv,
                background_images=backgrounds,
                pole_image=pole_rgba,
                yolo_weights=yolo_wts,
                img_size=(640, 640),

                # episodes
                steps_per_episode=steps_per_episode,

                # blobs & UV paint pairs
                count_max=80,
                area_cap=0.20,
                uv_paints=uv_paints,              # <— list of UVPaints
                default_uv_paint=uv_paints[0],    # harmless fallback

                # reward shaping (Phase B)
                eps_day_tolerance=eps_day_tolerance,
                day_floor=day_floor,

                # curriculum (Phase A)
                attack_only=attack_only,
                attack_alpha=attack_alpha,
            )
        )
    return _init


def parse_args():
    ap = argparse.ArgumentParser(description="Train PPO on UV-adv stop-sign env")
    ap.add_argument("--mode", choices=["attack", "uv", "both"], default="both",
                    help="attack=Phase A only, uv=Phase B only, both=Phase A then B")
    ap.add_argument("--steps-a", type=int, default=400_000, help="timesteps for Phase A (attack only)")
    ap.add_argument("--steps-b", type=int, default=1_200_000, help="timesteps for Phase B (UV gap)")
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    ap.add_argument("--ep1", type=int, default=12, help="episode steps in Phase A")
    ap.add_argument("--ep2", type=int, default=16, help="episode steps in Phase B")
    ap.add_argument("--attack-alpha", type=float, default=1.0, help="opacity for Phase A blobs")
    ap.add_argument("--yolo", default="./weights/yolo11n.pt")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--tb", default="./runs/tb")
    ap.add_argument("--ckpt", default="./runs/checkpoints")
    ap.add_argument("--overlays", default="./runs/overlays")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using cuda device")

    # --- Paths ---
    STOP_PLAIN = os.path.join(args.data, "stop_sign.png")
    STOP_UV    = os.path.join(args.data, "stop_sign_uv.png")
    POLE_PNG   = os.path.join(args.data, "pole.png")
    BG_DIR     = args.bgdir
    YOLO_WTS   = args.yolo

    # --- Load images (keep alpha for sign & pole) ---
    stop_plain = Image.open(STOP_PLAIN).convert("RGBA")
    if os.path.exists(STOP_UV):
        stop_uv = Image.open(STOP_UV).convert("RGBA")
    else:
        print(f"⚠️  {STOP_UV} not found — using stop_sign.png for UV phase.")
        stop_uv = stop_plain.copy()
    pole_rgba  = Image.open(POLE_PNG).convert("RGBA")
    backgrounds = load_backgrounds(BG_DIR)

    # You can add more pairs here:
    UV_PAINTS = [VIOLET_GLOW]  # e.g., [VIOLET_GLOW, GREEN_GLOW, BLUE_GLOW]

    # --- Env: initial phase config based on mode ---
    if args.mode == "attack":
        phaseA_cfg = dict(ep=args.ep1, attack=True)
        phaseB_cfg = None
    elif args.mode == "uv":
        phaseA_cfg = dict(ep=args.ep2, attack=False)
        phaseB_cfg = None
    else:  # both
        phaseA_cfg = dict(ep=args.ep1, attack=True)
        phaseB_cfg = dict(ep=args.ep2, attack=False)

    # --- Build vectorized env for current phase ---
    def build_env_for_phase(ep, attack):
        return VecTransposeImage(
            DummyVecEnv([
                make_env_factory(
                    stop_plain, stop_uv, pole_rgba, backgrounds, YOLO_WTS,
                    steps_per_episode=ep,
                    attack_only=attack, attack_alpha=args.attack_alpha,
                    uv_paints=UV_PAINTS,
                    eps_day_tolerance=0.03, day_floor=0.80,
                ) for _ in range(args.num_envs)
            ])
        )

    env = build_env_for_phase(phaseA_cfg["ep"], phaseA_cfg["attack"])

    # --- PPO setup ---
    N_STEPS = args.n_steps
    SAVE_FREQ = 2 * N_STEPS * args.num_envs  # every ~2 PPO updates

    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.tb, exist_ok=True)
    os.makedirs(args.overlays, exist_ok=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=2,
        n_steps=N_STEPS,
        batch_size=128,
        learning_rate=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.1,
        tensorboard_log=args.tb,
        device="auto",
    )

    # --- Callbacks (shared) ---
    tb_cb = TensorboardOverlayCallback(
        log_dir=args.tb,
        tag_prefix="uv_adv",
        max_images=50,
        verbose=1,   # print when it writes images/scalars
    )
    saver = SaveImprovingOverlaysCallback(
        save_dir=args.overlays,
        threshold=0.0,    # log from the start (no improvement gate)
        mode="adversary", # force adversary mode for early testing
        max_saved=50,
        verbose=1,        # print when saving/replacing/logging
        tb_callback=tb_cb,
    )

    # ----------------- Phase A -----------------
    if args.mode in ("attack", "both"):
        total_A = args.steps_a
        print(f"\n=== Phase A (attack-only) for {total_A:,} steps ===")
        ckptA = CheckpointCallback(save_freq=SAVE_FREQ, save_path=args.ckpt, name_prefix="phaseA")
        progressA = ProgressETACallback(total_timesteps=total_A, log_every_sec=15, verbose=1)
        model.learn(
            total_timesteps=total_A,
            callback=CallbackList([tb_cb, saver, ckptA, progressA]),
            tb_log_name="phaseA",
        )

    # ----------------- Switch to Phase B -----------------
    if phaseB_cfg:
        print("\nSwitching to Phase B (UV gap objective)...")
        env.set_attr("attack_only", False)
        env.set_attr("steps_per_episode", phaseB_cfg["ep"])

        total_B = args.steps_b
        ckptB = CheckpointCallback(save_freq=SAVE_FREQ, save_path=args.ckpt, name_prefix="phaseB")
        progressB = ProgressETACallback(total_timesteps=total_B, log_every_sec=15, verbose=1)
        model.learn(
            total_timesteps=total_B,
            reset_num_timesteps=False,
            callback=CallbackList([tb_cb, saver, ckptB, progressB]),
            tb_log_name="phaseB",
        )

    # --- Final save ---
    final_path = os.path.join(args.ckpt, "ppo_uv_adv_final")
    model.save(final_path)
    print(f"✅ Saved final model to {final_path}")
