# Stop Sign Grid UV Adversarial Training (YOLO + PPO)

This project trains a PPO agent to place small grid-cell overlays on a stop sign so that
YOLO confidence drops under UV activation while staying high in daylight. The environment
renders a sign-on-pole against randomized backgrounds with matched transforms, and uses
UV paint pairs (day vs UV-on) to model activation.

Ethics notice: this repository is for research and robustness testing only. Do not use it
to cause harm or unsafe behavior.

---

## Overview

Core ideas:
- Grid-cell action space on a stop sign octagon mask (discrete actions).
- UV paint pair: daylight color/alpha vs UV-on color/alpha.
- Matched transforms and backgrounds across daylight/UV variants for fair comparison.
- Reward that targets UV confidence drop while penalizing daylight drop and patch area.
- Early termination on success or area cap.

---

## Requirements

- Python 3.10+ recommended.
- PyTorch and stable-baselines3 (via requirements).
- YOLO weights in `weights/` (see below).

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Alternate: `enviornment.yml` is included if you prefer conda.

---

## Data and Weights

Required files in `data/`:
- `stop_sign.png` (RGBA, transparent background).
- `pole.png` (RGBA).
- `backgrounds/` (folder with scene images; 640x640 recommended).

Optional:
- `stop_sign_uv.png` (RGBA UV-lit version of the sign; if missing, the base sign is reused).

YOLO weights go in `weights/`:
- `weights/yolo11n.pt` (default)
- `weights/yolo8n.pt` (optional if you switch versions)

---

## Quick Start

Minimal run:

```bash
python train_single_stop_sign.py --data ./data --bgdir ./data/backgrounds
```

Resume from latest checkpoint:

```bash
python train_single_stop_sign.py --resume
```

Use a specific YOLO version/weights:

```bash
python train_single_stop_sign.py --yolo-version 11 --yolo-weights ./weights/yolo11n.pt
```

---

## Key Training Flags

From `train_single_stop_sign.py`:

- `--num-envs` (default 8) and `--vec` (`dummy` or `subproc`)
- `--n-steps`, `--batch-size`, `--total-steps` (PPO training control)
- `--episode-steps` (max steps per episode)
- `--grid-cell` (2, 4, 8, 16, 32) grid size in pixels
- `--uv-threshold` UV drop threshold for success
- `--lambda-area` area penalty strength (encourages minimal patches)
- `--area-cap-frac` cap on total patch area (<= 0 disables)
- `--area-cap-penalty` reward penalty when cap would be exceeded
- `--detector-device` (e.g., `cpu`, `cuda`, or `auto`)
- `--ckpt`, `--overlays`, `--tb` output paths
- `--save-freq-steps` or `--save-freq-updates` checkpoint cadence

---

## Environment Details

The environment is implemented in `envs/stop_sign_grid_env.py`.

Highlights:
- Discrete action space over valid grid cells inside the sign octagon.
- UV-on reward uses smoothed UV drop (`drop_on_smooth`) to reduce noise.
- Area cap supports soft (penalty) or hard (terminate) modes.
- Minimum UV alpha (`uv_min_alpha`) ensures patches are visible under UV even with
  very low paint alpha.

If you need to change rendering or physics:
- `_transform_sign()` controls camera jitter, blur, color, and noise.
- `_compose_sign_and_pole()` controls pole ratio and placement.
- `_place_group_on_background()` controls scale and background placement.

---

## Logging and Metrics

TensorBoard logs:

```bash
tensorboard --logdir runs/tb --port 6006
```

Callbacks log:
- `TensorboardOverlayCallback` (overlay images and metadata)
- `EpisodeMetricsCallback` (episode-end scalars)

Episode metrics currently include:
- `episode/area_frac_final`, `episode/length_steps`
- `episode/drop_on_final`, `episode/drop_on_smooth_final`
- `episode/base_conf_final`, `episode/after_conf_final`
- `episode/reward_final`, `episode/selected_cells_final`
- `episode/eval_K_used_final`
- `episode/uv_success_final`, `episode/area_cap_exceeded_final`

---

## Output Artifacts

Generated files:
- `runs/checkpoints/` PPO checkpoints.
- `runs/overlays/` best overlays (PNG + JSON) and `traces.ndjson`.
- `runs/tb/` TensorBoard event files.

Overlay saver:
- `utils/save_callbacks.py` keeps the best N overlays and appends trace metadata.
- Current training config keeps the top 1000 minimal-area successes.
- Files are named by area fraction and step, for example:
  - `area0p1234_step000000123_env00_full.png`
  - `area0p1234_step000000123_env00_overlay.png`
  - `area0p1234_step000000123_env00.json`

Trace replay:
- `trace_replay.py` can rebuild a mask or full composite from trace rows.

---

## Debugging and Tools

- `tools/debug_grid_env.py` runs the env step-by-step and saves UV-on previews.
- `tools/cleanup_runs.py` removes old run outputs (dry-run by default).
- `preview_stop_sign_3d.sh` renders a quick 3D preview (if your setup includes it).
- `setup_env.sh` contains a helper for local setup.

Cleanup usage:
```bash
# Dry-run
python tools/cleanup_runs.py

# Delete
python tools/cleanup_runs.py --yes
```

---

## Directory Structure

```
.
|-- data/
|   |-- stop_sign.png
|   |-- stop_sign_uv.png
|   |-- pole.png
|   |-- backgrounds/
|
|-- detectors/
|   |-- yolo_wrapper.py
|
|-- envs/
|   |-- stop_sign_grid_env.py
|
|-- tools/
|   |-- debug_grid_env.py
|   |-- cleanup_runs.py
|
|-- utils/
|   |-- save_callbacks.py
|   |-- tb_callbacks.py
|   |-- uv_paint.py
|
|-- weights/
|   |-- yolo11n.pt
|   |-- yolo8n.pt
|
|-- runs/
|   |-- checkpoints/
|   |-- overlays/
|   |-- tb/
|
|-- train_single_stop_sign.py
|-- train.sh
|-- trace_replay.py
|-- requirements.txt
|-- enviornment.yml
```

---

## Tips

- If you run on CUDA, `--vec dummy` is safer with YOLO inference.
- Lower `grid-cell` and higher `lambda-area` tend to produce smaller patches.
- If you are not seeing UV drop, increase `eval-K` to reduce variance.

---

## License

MIT for research and educational use.
