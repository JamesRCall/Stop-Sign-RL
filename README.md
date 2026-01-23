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
- Efficiency bonus (drop per area) and optional adaptive area penalty to favor minimal patches.
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
- `weights/yolo8n.pt` (default)
- `weights/yolo11n.pt` (optional if you switch versions)

---

## Quick Start

Minimal run:

```bash
python train_single_stop_sign.py --data ./data --bgdir ./data/backgrounds
```

Recommended single-machine run:

```bash
bash train.sh --yolo-weights ./weights/yolo8n.pt --multiphase
```

Resume from latest checkpoint:

```bash
python train_single_stop_sign.py --resume
```

Use a specific YOLO version/weights:

```bash
python train_single_stop_sign.py --yolo-version 8 --yolo-weights ./weights/yolo8n.pt
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
- `--lambda-efficiency` efficiency bonus (drop per area)
- `--area-target` (default 0.20) and `--area-lagrange-lr/min/max` adaptive area penalty target and bounds
- `--step-cost` (default 0.0) and `--step-cost-after-target` (default 0.03) per-step penalties
- `--step-cost`, `--step-cost-after-target` per-step penalties
- `--lambda-area-start`, `--lambda-area-end`, `--lambda-area-steps` (curriculum)
- `--area-cap-frac` cap on total patch area (<= 0 disables)
- `--area-cap-penalty` reward penalty when cap would be exceeded
- `--area-cap-mode` (`soft` or `hard`)
- `--area-cap-start`, `--area-cap-end`, `--area-cap-steps` (curriculum)
- `--lambda-day` penalty for daylight confidence drop beyond tolerance
- `--lambda-iou`, `--lambda-misclass` extra objectives for mislocalization/misclassification
- `--paint`, `--paint-list` paint selection (single or per-episode sampling)
- `--multiphase` enable 3-phase curriculum (solid/no pole -> dataset + pole)
- `--phase1-steps`, `--phase2-steps`, `--phase3-steps` (phase lengths; 0 = auto split)
- `--phase1-eval-K`, `--phase2-eval-K`, `--phase3-eval-K` (per-phase eval_K overrides)
- `--phase1-step-cost`, `--phase2-step-cost`, `--phase3-step-cost` (phase overrides)
- `--bg-mode` (`dataset` or `solid`) and `--no-pole` for single-phase
- `--obs-size`, `--obs-margin`, `--obs-include-mask` (cropped observation + mask channel)
- `--ent-coef`, `--ent-coef-start`, `--ent-coef-end`, `--ent-coef-steps` (entropy coefficient schedule)
- `--detector-device` (e.g., `cpu`, `cuda`, or `auto`)
- `--step-log-every`, `--step-log-keep`, `--step-log-500` (step logging control)
- `--ckpt`, `--overlays`, `--tb` output paths (TB logs grouped under `grid_uv_yolo<ver>`)
- `--save-freq-steps` or `--save-freq-updates` checkpoint cadence

---

## Environment Details

The environment is implemented in `envs/stop_sign_grid_env.py`.

Highlights:
- Discrete action space over valid grid cells inside the sign octagon.
- UV-on reward uses raw UV drop (`drop_on`) computed as the day baseline
  confidence minus UV-on overlay confidence.
- Reward includes an efficiency bonus (drop per area) and an optional adaptive
  area penalty that nudges toward a target patch fraction.
- Optional per-step penalties can apply globally or only after the area target.
- Observations are cropped around the sign with an optional overlay-mask channel
  (controlled by `--obs-*` flags).
- Training uses a lightweight custom CNN extractor tuned for sign crops.
- Area cap supports soft (penalty) or hard (terminate) modes.
- Minimum UV alpha (`uv_min_alpha`) ensures patches are visible under UV even with
  very low paint alpha.

### Reward Equation (current)

Definitions:
- `c0_day`: baseline day confidence (no overlay)
- `c_day`: day confidence with overlay
- `c_on`: UV-on confidence with overlay
- `drop_day = c0_day - c_day`
- `drop_on = c0_day - c_on`
- `area = total_area_mask_frac`
- `mean_iou`: mean IoU between target box and top detection
- `misclass`: misclassification rate

Adaptive area penalty:
- `lambda_area_used = lambda_area` by default
- If `area_lagrange_lr > 0` and `area_target` is set:
  - `lambda_area_dyn = clip(lambda_area_dyn + area_lagrange_lr * (area - area_target),
    area_lagrange_min, area_lagrange_max)`
  - `lambda_area_used = lambda_area_dyn`

Efficiency bonus:
```
eff = log1p(max(0, drop_on) / max(area, efficiency_eps))
```

Core reward:
```
pen_day  = max(0, drop_day - day_tolerance)
raw_core = drop_on
         - lambda_day * pen_day
         - lambda_area_used * area
         + lambda_iou * (1 - mean_iou)
         + lambda_misclass * misclass
         + lambda_efficiency * eff
         - lambda_perceptual * perceptual_delta
```

Shaping + success:
```
shaping       = 0.35 * tanh(3.0 * (success_conf - c_on))
success_bonus = 0.2 if c_on <= success_conf and within_cap else 0
raw_total     = raw_core + shaping + success_bonus
```

Soft cap override (if enabled and exceeded):
```
excess    = max(0, (area - area_cap) / area_cap)
over_pen  = abs(area_cap_penalty) * (1 + 2 * excess)
raw_total = -over_pen
```

Final reward:
```
reward = tanh(1.2 * raw_total)
```

If you need to change rendering or physics:
- `_transform_sign()` controls camera jitter, blur, color, and noise.
- `_compose_sign_and_pole()` controls pole ratio and placement.
- `_place_group_on_background()` controls scale and background placement.

---

## Logging and Metrics

TensorBoard logs:

```bash
# train_single_stop_sign.py
tensorboard --logdir runs/tb --port 6006
# train.sh (defaults)
tensorboard --logdir _runs/tb --port 6006
```

Callbacks log:
- `TensorboardOverlayCallback` (overlay images and metadata)
- `EpisodeMetricsCallback` (episode-end scalars)
- `StepMetricsCallback` (rolling step metrics)

Episode metrics currently include:
- `episode/area_frac_final`, `episode/length_steps`
- `episode/drop_on_final`, `episode/drop_on_smooth_final`
- `episode/base_conf_final`, `episode/after_conf_final`
- `episode/reward_final`, `episode/selected_cells_final`
- `episode/eval_K_used_final`
- `episode/uv_success_final`, `episode/area_cap_exceeded_final`
- `episode/reward_core_final`, `episode/reward_raw_total_final`
- `episode/reward_efficiency_final`, `episode/reward_perceptual_final`
- `episode/lambda_area_used_final`, `episode/lambda_area_dyn_final`
- `episode/area_target_frac_final`, `episode/area_lagrange_lr_final`
- `episode/area_reward_corr` (rolling correlation between area and reward)

Step metrics:
- Rolling window of per-step rows in
  `runs/tb/grid_uv_yolo8/<phase>/tb_step_metrics/step_metrics.ndjson`
- 500-step snapshots in
  `runs/tb/grid_uv_yolo8/<phase>/tb_step_metrics/step_metrics_500.ndjson`
- Step scalars include reward components and adaptive area weights when present.

---

## Output Artifacts

Generated files:
- `runs/checkpoints/` PPO checkpoints.
- `runs/overlays/` best overlays (PNG + JSON) and `traces.ndjson`.
- `runs/tb/` TensorBoard event files (grouped under `grid_uv_yolo<ver>/<phase>`).
- If you use `train.sh` defaults, outputs go to `_runs/` instead of `runs/`.
  TensorBoard logs are grouped under `_runs/tb/grid_uv_yolo<ver>/<phase>`.

Overlay saver:
- `utils/save_callbacks.py` keeps the best N overlays and appends trace metadata.
- Current training config keeps the top 1000 minimal-area successes.
- Files are named by area fraction and step, for example:
  - `area0p1234_step000000123_env00_full.png`
  - `area0p1234_step000000123_env00_overlay.png`
  - `area0p1234_step000000123_env00.json`

Trace replay:
- Removed (legacy blob traces no longer apply to the grid environment).

---

## Debugging and Tools

- `tools/debug_grid_env.py` runs the env step-by-step and saves UV-on previews.
- `tools/area_sweep_debug.py` sweeps coverage levels and logs confidence/IoU/misclass stats.
- `tools/replay_area_sweep.py` replays logged sweep cases and saves images.
- `tools/test_stop_sign_confidence.py` checks detector confidence on a single image.
- `tools/cleanup_runs.py` removes old run outputs (dry-run by default).
- `tools/detector_server.py` runs a shared YOLO detector for multi-process training.
- `setup_env.sh` contains a helper for local setup.

Cleanup usage:
```bash
# Dry-run
python tools/cleanup_runs.py

# Delete
python tools/cleanup_runs.py --yes
```

Detector server usage:
```bash
python tools/detector_server.py --model ./weights/yolo8n.pt --device cuda:0 --port 5009

# In training, point the detector device to the server:
# --detector-device server://HOST:5009
```

Single-command server + training (from `train.sh`):
```bash
bash train.sh --yolo-version 8 --yolo-weights ./weights/yolo8n.pt --start-detector-server
```

Common single-machine training (no server):
```bash
bash train.sh --yolo-version 8 --yolo-weights ./weights/yolo8n.pt
```

Important `train.sh` knobs:
- `--num-envs`, `--vec`: number of envs and vectorization mode; use `--vec dummy` with GPU YOLO.
- `--n-steps`, `--batch`, `--total-steps`: PPO rollout size, batch size, and total training steps.
- `--grid-cell`: patch grid size in pixels (2, 4, 8, 16, 32).
- `--uv-threshold`: UV drop threshold for success.
- `--lambda-area`, `--lambda-area-start/end/steps`: area penalty and optional ramp.
- `--area-cap-frac`, `--area-cap-mode`: patch area cap and soft/hard behavior.
- `--area-cap-start/end/steps`: cap curriculum from larger to smaller.
- `--obs-size`, `--obs-margin`, `--obs-include-mask`: observation crop and mask channel.
- `--ent-coef`, `--ent-coef-start/end/steps`: entropy coefficient schedule.
- `--step-log-every`, `--step-log-keep`, `--step-log-500`: step metrics logging controls.

---

## Commenting Guidelines

- Keep comments sparse and focused on *why* a block exists or what it protects against.
- Avoid restating obvious code; prefer naming and structure to make intent clear.
- When behavior is non-obvious (curriculum logic, reward shaping), add a short note.

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
|   |-- area_sweep_debug.py
|   |-- replay_area_sweep.py
|   |-- test_stop_sign_confidence.py
|   |-- cleanup_runs.py
|   |-- detector_server.py
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
