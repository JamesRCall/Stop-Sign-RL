# train_single_stop_sign.py
import os
import time
from PIL import Image
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList, BaseCallback, CheckpointCallback
)

from envs.stop_sign_env import StopSignBlobEnv
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback


# === ETA Progress Display ===
class ProgressETACallback(BaseCallback):
    def __init__(self, total_timesteps: int, log_every_sec: float = 10.0, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.log_every_sec = float(log_every_sec)
        self._t0 = None
        self._last_log_t = None

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()
        self._last_log_t = self._t0

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if now - self._last_log_t >= self.log_every_sec:
            done = self.num_timesteps
            elapsed = now - self._t0
            sps = done / max(elapsed, 1e-9)
            rem_steps = max(self.total_timesteps - done, 0)
            eta_sec = rem_steps / max(sps, 1e-9)
            if self.verbose:
                print(f"[Progress] {done:,}/{self.total_timesteps:,} "
                      f"({done/self.total_timesteps:6.2%}) | "
                      f"{sps:,.1f} steps/s | ETA ~ {eta_sec/3600:,.2f} h")
            self.logger.record("progress/steps_per_sec", sps)
            self.logger.record("progress/eta_hours", eta_sec / 3600.0)
            self._last_log_t = now
        return True


# === Watchdog (auto-save + exit if stuck) ===
class WatchdogRestartCallback(BaseCallback):
    def __init__(self, save_path, check_interval_sec=1800, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_interval_sec = check_interval_sec
        self.last_step_time = None
        self.last_step_count = 0

    def _on_training_start(self):
        self.last_step_time = time.time()
        self.last_step_count = self.num_timesteps

    def _on_step(self):
        now = time.time()
        if now - self.last_step_time > self.check_interval_sec:
            if self.num_timesteps == self.last_step_count:
                print("⚠️  No progress detected in 30+ min. Saving checkpoint and exiting...")
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                self.model.save(self.save_path)
                raise SystemExit("Training stalled; checkpoint saved.")
            else:
                self.last_step_count = self.num_timesteps
                self.last_step_time = now
        return True


if __name__ == "__main__":
    print("torch.cuda.is_available():", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", device, "device")

    # === Load data ===
    # keep alpha so the env can build the mask from it
    stop_img = Image.open("./data/stop_sign.png")
    yolo_weights = "./weights/yolo11n.pt"  # YOLOv11n weights

    # === Phase configs ===
    PHASE1 = dict(K_transforms=8,  w_area=1.0, w_count=0.05)   # fast
    PHASE2 = dict(K_transforms=16, w_area=2.0, w_count=0.10)   # robust

    # === Env factory (Phase 1 defaults) ===
    def make_env():
        def _init():
            return Monitor(
                StopSignBlobEnv(
                    stop_sign_image=stop_img,
                    yolo_weights=yolo_weights,
                    img_size=(640, 640),
                    K_transforms=PHASE1["K_transforms"],
                    count_max=80,
                    elite_bonus_weight=0.0,  # ignored by env now
                    elite_momentum=0.0,      # ignored by env now
                    area_cap=0.25,
                    w_area=PHASE1["w_area"],
                    w_count=PHASE1["w_count"]
                )
            )
        return _init

    # === Vectorized env ===
    NUM_ENVS = 8  # adjust if too heavy
    env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])
    env = VecTransposeImage(env)  # (H,W,C) -> (C,H,W) for CNN policy

    # === PPO hyperparams ===
    N_STEPS = 512  # rollout length per env
    # One PPO update uses N_STEPS * NUM_ENVS env-steps.
    # We want a checkpoint every TWO PPO updates:
    SAVE_FREQ = 2 * N_STEPS * NUM_ENVS  # = 2 * 512 * NUM_ENVS

    # === Checkpoint resume ===
    os.makedirs("./runs/checkpoints", exist_ok=True)
    checkpoint_path = "./runs/checkpoints/ppo_watchdog_latest.zip"
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint:", checkpoint_path)
        model = PPO.load(checkpoint_path, env=env, device="auto")
    else:
        print("Starting new PPO model...")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=2,
            n_steps=N_STEPS,
            batch_size=128,
            device="auto",
            tensorboard_log="./runs/ppo_adversary"
        )

    # === Callbacks ===
    tb_cb = TensorboardOverlayCallback(
        log_dir="./runs/ppo_adversary", tag_prefix="adv", max_images=200, verbose=0
    )
    saver = SaveImprovingOverlaysCallback(
        "./runs/overlays_adversary",
        threshold=0.02,  # require ≥0.02 improvement to save
        mode="auto",     # use env objective (adversary/helper)
        max_saved=300,
        verbose=1,
        tb_callback=tb_cb,
    )
    watchdog = WatchdogRestartCallback("./runs/checkpoints/ppo_watchdog_latest.zip")

    # ✅ Checkpoint every TWO PPO updates (computed above)
    ckpt_cb_phase1 = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path="./runs/checkpoints",
        name_prefix="ppo_phase1"
    )
    ckpt_cb_phase2 = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path="./runs/checkpoints",
        name_prefix="ppo_phase2"
    )

    # === Phase 1: Fast ===
    phase1_steps = 500_000
    p1_callbacks = CallbackList([
        saver,
        tb_cb,
        ProgressETACallback(total_timesteps=phase1_steps, log_every_sec=15, verbose=1),
        ckpt_cb_phase1,
        watchdog
    ])
    model.learn(total_timesteps=phase1_steps, callback=p1_callbacks, tb_log_name="PPO_phase1")

    # === Switch env hyperparams to robust ===
    print("\nSwitching to Phase 2: robust settings...")
    env.set_attr("K", PHASE2["K_transforms"])
    env.set_attr("w_area", PHASE2["w_area"])
    env.set_attr("w_count", PHASE2["w_count"])

    # === Phase 2: Robust ===
    phase2_steps = 1_500_000
    p2_callbacks = CallbackList([
        saver,
        tb_cb,
        ProgressETACallback(total_timesteps=phase2_steps, log_every_sec=15, verbose=1),
        ckpt_cb_phase2,
        watchdog
    ])
    model.learn(
        total_timesteps=phase2_steps,
        reset_num_timesteps=False,
        callback=p2_callbacks,
        tb_log_name="PPO_phase2",
    )

    # === Final checkpoint ===
    model.save("./runs/checkpoints/ppo_adversary_final")
    print("✅ Saved final model to ./runs/checkpoints/ppo_adversary_final")
