import os, glob, time, argparse
from typing import List, Optional
from PIL import Image
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from envs.stop_sign_grid_env import StopSignGridEnv
from utils.uv_paint import GREEN_GLOW  # single pair; swap here if you want another
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback


# ----------------- progress logger -----------------
class ProgressETACallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every_sec: float = 10.0, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.log_every_sec = float(log_every_sec)
        self._t0 = None
        self._last = None

    def _on_training_start(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if (now - self._last) >= self.log_every_sec:
            done = self.num_timesteps
            sps = done / max(now - self._t0, 1e-9)
            if self.verbose:
                eta = max(self.model._total_timesteps - done, 0) / max(sps, 1e-9)
                print(f"[{done:,}/{self.model._total_timesteps:,}] {sps:,.2f} steps/s | ETA {eta/3600:,.2f}h")
            self.logger.record("progress/steps_per_sec", sps)
            self._last = now
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


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def make_env_factory(
    stop_plain: Image.Image,
    stop_uv: Image.Image,
    pole_rgba: Image.Image,
    backgrounds: List[Image.Image],
    steps_per_episode: int,
    eval_K: int,
    grid_cell_px: int,
    uv_drop_threshold: float,
    yolo_wts: str,
    yolo_device: str,
):
    def _init():
        return Monitor(
            StopSignGridEnv(
                stop_sign_image=stop_plain,
                stop_sign_uv_image=stop_uv,
                background_images=backgrounds,
                pole_image=pole_rgba,
                yolo_weights=yolo_wts,
                yolo_device=yolo_device,
                img_size=(640, 640),

                steps_per_episode=steps_per_episode,
                eval_K=eval_K,
                detector_debug=True,

                grid_cell_px=grid_cell_px,
                max_cells=None,  # optional cap; leave None because we terminate by threshold
                uv_paint=VIOLET_GLOW,  # single color pair for this project
                use_single_color=True,

                uv_drop_threshold=uv_drop_threshold,
                day_tolerance=0.05,
                lambda_day=1.0,
            )
        )
    return _init


def parse_args():
    ap = argparse.ArgumentParser("Train PPO on grid-square UV attack over stop sign")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--yolo", default="./weights/yolo11n.pt")
    ap.add_argument("--detector-device", default=os.getenv("YOLO_DEVICE", "auto"))
    ap.add_argument("--tb", default="./runs/tb")
    ap.add_argument("--ckpt", default="./runs/checkpoints")
    ap.add_argument("--overlays", default="./runs/overlays")

    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--total-steps", type=int, default=400_000)

    ap.add_argument("--episode-steps", type=int, default=7000)
    ap.add_argument("--eval-K", type=int, default=10)
    ap.add_argument("--grid-cell", type=int, default=2, choices=[2, 4])
    ap.add_argument("--uv-threshold", type=float, default=0.70)

    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --ckpt")

    ap.add_argument("--save-freq-steps", type=int, default=0)
    ap.add_argument("--save-freq-updates", type=int, default=2)
    return ap.parse_args()


if __name__ == "__main__":
    # allocator knobs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    if "cuda" in str(args.detector_device).lower() and args.vec == "subproc":
        print("⚠️ CUDA detector + SubprocVecEnv is risky. Switching vec to dummy.")
        args.vec = "dummy"
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using cuda device")

    

    # paths
    STOP_PLAIN = os.path.join(args.data, "stop_sign.png")
    STOP_UV    = os.path.join(args.data, "stop_sign_uv.png")
    POLE_PNG   = os.path.join(args.data, "pole.png")
    BG_DIR     = args.bgdir

    stop_plain = Image.open(STOP_PLAIN).convert("RGBA")
    stop_uv    = Image.open(STOP_UV).convert("RGBA") if os.path.exists(STOP_UV) else stop_plain.copy()
    pole_rgba  = Image.open(POLE_PNG).convert("RGBA")
    bgs        = load_backgrounds(BG_DIR)

    def build_env():
        fns = [
            make_env_factory(
                stop_plain, stop_uv, pole_rgba, bgs,
                steps_per_episode=args.episode_steps,
                eval_K=args.eval_K,
                grid_cell_px=args.grid_cell,
                uv_drop_threshold=args.uv_threshold,
                yolo_wts=args.yolo,
                yolo_device=args.detector_device,
            ) for _ in range(args.num_envs)
        ]
        v = SubprocVecEnv(fns) if args.vec == "subproc" else DummyVecEnv(fns)
        return VecTransposeImage(v)

    env = build_env()

    # PPO
    os.makedirs(args.tb, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.overlays, exist_ok=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=2,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        clip_range=0.1,
        tensorboard_log=args.tb,
        device="auto",
    )

    # resume if asked
    if args.resume:
        ckpt = find_latest_checkpoint(args.ckpt)
        if ckpt:
            print(f"🔄 Resuming from: {ckpt}")
            model = PPO.load(ckpt, env=env, device="auto")
            model.n_steps = int(args.n_steps)
            model.batch_size = int(args.batch_size)

    # callbacks
    # Save “best” (lowest after_conf) UV-on examples, keep top-50, log to TB
    tb_cb = TensorboardOverlayCallback(args.tb, tag_prefix="grid_uv", max_images=25, verbose=1)
    saver = SaveImprovingOverlaysCallback(
        save_dir=args.overlays, threshold=0.0, mode="adversary",
        max_saved=50, verbose=1, tb_callback=tb_cb
    )

    # checkpoint cadence
    if args.save_freq_steps and args.save_freq_steps > 0:
        SAVE_FREQ = int(args.save_freq_steps)
    else:
        SAVE_FREQ = max(int(args.n_steps) * int(args.num_envs) * max(args.save_freq_updates, 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=SAVE_FREQ, save_path=args.ckpt, name_prefix="grid")

    progress = ProgressETACallback(total_timesteps=int(args.total_steps), log_every_sec=15, verbose=1)

    model.learn(
        total_timesteps=int(args.total_steps),
        callback=CallbackList([tb_cb, saver, ckpt_cb, progress]),
        tb_log_name="grid_uv",
    )

    final = os.path.join(args.ckpt, "ppo_grid_uv_final")
    model.save(final)
    print(f"✅ Saved final model to {final}")
