# Minimal Adversarial Patches for Stop Signs (RL + YOLOv11)

This repo trains a PPO agent that paints **opaque blobs** on a stop sign image to **lower YOLOv11 detection confidence** while **penalizing total patch area** and **blob count**. It supports strong augmentations (angles, blur, JPEG artifacts, illumination), TensorBoard logging, periodic checkpoints, and a Top‑K overlay saver that keeps only the best examples on disk.

> **Ethics & scope:** This project is intended strictly for **adversarial robustness research** (e.g., testing defenses). Do not deploy to mislead people or cause harm.

---

## Table of Contents
- [Repo structure](#repo-structure)
- [Prerequisites](#prerequisites)
- [Install (Windows, macOS/Linux)](#install-windows-macoslinux)
- [YOLO weights (Git LFS)](#yolo-weights-git-lfs)
- [Quickstart](#quickstart)
- [Training phases & hyperparameters](#training-phases--hyperparameters)
- [Resuming from checkpoints](#resuming-from-checkpoints)
- [TensorBoard](#tensorboard)
- [Remote TensorBoard (SSH or ngrok)](#remote-tensorboard-ssh-or-ngrok)
- [Changing patch constraints](#changing-patch-constraints)
- [What gets saved & where](#what-gets-saved--where)
- [Troubleshooting & performance tips](#troubleshooting--performance-tips)
- [License](#license)
- [Citation](#citation)

---

## Repo structure

```
.
├─ data/
│  └─ stop_sign.png            # RGBA (transparent background recommended)
├─ weights/
│  └─ yolo11n.pt               # YOLOv11n weights (tracked with Git LFS)
├─ envs/
│  └─ stop_sign_env.py         # Gymnasium env: augmentations, reward, constraints
├─ detectors/
│  └─ yolo_wrapper.py          # YOLO wrapper (Ultralytics)
├─ utils/
│  ├─ save_callbacks.py        # Top‑K saver (keeps best overlays, evicts worst)
│  └─ tb_callbacks.py          # TensorBoard image + scalar logging
├─ runs/
│  ├─ checkpoints/             # PPO checkpoints (auto-created)
│  └─ overlays_adversary/      # Saved overlay PNG/JSON (Top‑K)
├─ train_single_stop_sign.py   # Main training script (two-phase curriculum)
├─ requirements.txt
├─ environment.yml  
└─ README.md
```

---

## Prerequisites
- **Python 3.10–3.12** recommended
- **CUDA-capable GPU** (optional but strongly recommended)  
- OS: Windows 10/11, macOS 12+, or Linux

> If you use Windows + NVIDIA GPU, install the latest **NVIDIA driver** and a matching **CUDA runtime** (PyTorch wheels often bundle CUDA).

---

## Install (Windows, macOS/Linux)

### Option A — *pip + venv (recommended)*

#### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — *Conda*
```bash
conda env create -f environment.yml   # if you keep an env file
conda activate stop-sign-rl
```

> You **should activate the venv/conda env before running** any of the commands below.

---

## YOLO weights (Git LFS)

Large binaries (e.g., `weights/yolo11n.pt`) should be stored with **Git LFS** so your repo stays lean.

Install & initialize LFS **once**:
```bash
git lfs install
git lfs track "weights/*.pt"
git add .gitattributes
git add weights/yolo11n.pt
git commit -m "Track YOLO weights with LFS"
git push
```

> To use a different YOLO model, place it in `weights/` and update `yolo_weights` in `train_single_stop_sign.py`.

---

## Quickstart

1) Put a **transparent-background** stop sign in `data/stop_sign.png` (RGBA is best).  
   The env uses the alpha channel to isolate the sign (including white border & text).

2) Put YOLO weights in `weights/yolo11n.pt` (or your preferred `.pt`).

3) Train:
```bash
python train_single_stop_sign.py
```

- Logs go to `./runs/ppo_adversary/` (for TensorBoard)
- Checkpoints go to `./runs/checkpoints/`
- Top‑K overlays go to `./runs/overlays_adversary/`

---

## Training phases & hyperparameters

`train_single_stop_sign.py` runs **two phases**:

- **Phase 1 (fast)**: lower `K_transforms` for speed (`K=8` by default), milder penalties.  
- **Phase 2 (robust)**: higher `K` (`K=16`) for stronger robustness and higher area/count penalties.

Key knobs (defined when constructing `StopSignBlobEnv` in the script or directly in `envs/stop_sign_env.py`):
- `count_max`: upper bound for # of blobs per step
- `area_cap`: fraction of the sign area the **axis-aligned bounding box** of blobs may cover
- `w_area`, `w_count`: strength of penalties for area and blob count
- `K_transforms`: number of random augmentations used to measure YOLO confidence
- `min_count_penalty`, `min_area_frac`, `min_area_penalty`: **anti-collapse** terms to discourage degenerate single tiny blob solutions

The reward is **normalized** per-step relative to the current base confidence, so PPO sees more consistent scales across different images and training stages.

---

## Resuming from checkpoints

The script automatically looks for `./runs/checkpoints/ppo_watchdog_latest.zip` and resumes if present.

Manual resume:
```bash
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# ... build env same way as in train_single_stop_sign.py ...

model = PPO.load("./runs/checkpoints/ppo_watchdog_latest.zip", env=env, device="auto")
model.learn(total_timesteps=100_000, reset_num_timesteps=False)
```

> **Tip:** The script also writes rolling phase checkpoints every **~16k–20k env steps** (configurable), so you can restore to the nearest one after interruptions.

---

## TensorBoard

Start it in another terminal (with your venv active):
```bash
tensorboard --logdir ./runs/ppo_adversary --port 6006
```

Open http://localhost:6006/

Scalars to watch:
- `adv/base_conf`, `adv/after_conf`, `adv/delta_conf`
- `progress/steps_per_sec`, `progress/eta_hours`
- `adv/count`, `adv/size_scale`, `adv/total_patch_area_frac`

Images:
- `adv/overlay_img` shows recent composited stop signs with blobs.

---

## Remote TensorBoard (SSH or ngrok)

**SSH port forward (best if same LAN/VPN):**
```bash
ssh -L 6006:localhost:6006 user@your_training_pc
# then open http://localhost:6006 on your laptop/phone
```

**ngrok (simple tunnel):**
```bash
ngrok http 6006
# use the https://*.ngrok-free.app URL it prints
```
> Only share the URL with people you trust. Consider enabling basic auth via ngrok config.

---

## Changing patch constraints

Open **`envs/stop_sign_env.py`** and edit defaults in `StopSignBlobEnv.__init__` or override them in the constructor within `train_single_stop_sign.py`:

```python
StopSignBlobEnv(
    # ...
    count_max=60,        # lower the max number of blobs
    area_cap=0.20,       # tighten max spread (AABB of overlays / sign area)
    w_area=2.0,          # stronger area penalty
    w_count=0.15,        # stronger count penalty
    K_transforms=16,     # more robust eval, slower
    min_count_penalty=0.15,
    min_area_frac=0.01,
    min_area_penalty=0.5,
)
```

> Blobs are **opaque** (simulating paint). Colors are **single-color per step** (can change across steps).

---

## What gets saved & where

- **Checkpoints**: `./runs/checkpoints/`
  - Rolling *watchdog* checkpoint: `ppo_watchdog_latest.zip` (updated periodically & on stall)
  - Phase checkpoints (e.g., `ppo_phase1_00020000_steps.zip`)

- **Overlays (Top‑K)**: `./runs/overlays_adversary/`
  - `overlay_XXXXX.png` + `overlay_XXXXX.json`
  - The saver keeps only the best **K** (default `max_saved=300`), evicting the current worst to cap disk usage.

- **TensorBoard logs**: `./runs/ppo_adversary/`

---

## Troubleshooting & performance tips

- **Windows multiprocessing stalls**: start with `DummyVecEnv` (already used here). If you switch to `SubprocVecEnv`, try fewer envs and set `spawn` start method.
- **Too slow**: reduce `K_transforms`, lower image size (e.g., 512), reduce augmentations, use fewer parallel envs, or a lighter YOLO model.
- **GPU underutilized**: increase `num_envs` gradually (watch VRAM in `nvidia-smi`), or raise `n_steps`.
- **No checkpoints**: make sure the `CheckpointCallback` is **added to the CallbackList** and that `save_freq` is not larger than your total steps for that phase. Check write permissions and free disk space.
- **Masking**: with RGBA `stop_sign.png`, the env builds the mask from **alpha** (so background stays excluded). If you switch to RGB, the fallback color-thresholding can include red backgrounds—prefer RGBA.

---

## License

MIT.

---

## Citation

TODO

```bibtex
@software{stop_sign_minimal_patches,
  title = {Minimal Adversarial Patches for Stop Signs (RL + YOLOv11)},
  author = {TODO},
  year = {2025},
  url = {TODO}
}
```
