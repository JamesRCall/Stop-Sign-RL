# Stop Sign Grid UV Adversarial Training (PPO + Multi-Detector)

This project trains a PPO agent to place small grid-cell overlays on a stop sign so that
detector confidence drops under UV activation while staying high in daylight. The environment
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
- Efficiency bonus (drop per area) and fixed area penalties to favor minimal patches.
- Early termination on success or area cap.
- Detector backends: Ultralytics YOLO, torchvision detectors, and optional Transformers DETR.

---

## Requirements

- Python 3.10+ recommended.
- PyTorch, stable-baselines3, and sb3-contrib (MaskablePPO).
- YOLO weights in `weights/` (see below).
- Optional: `transformers` if you use the DETR backend.

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Alternate: `enviornment.yml` is included if you prefer conda.

If you installed requirements before action masking was added, you may need:

```bash
python -m pip install sb3-contrib
```

Optional (DETR backend):
```bash
python -m pip install transformers
```

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

Torchvision detectors download pretrained weights automatically on first use
(cached under `~/.cache/torch/hub/checkpoints`).

Transformers DETR models download from Hugging Face on first use
(cached under `~/.cache/huggingface`).

---

## Quick Start

Minimal run (train.sh defaults):

```bash
bash train.sh
```

Recommended single-machine run (YOLOv8, GPU, dummy vec):

```bash
YOLO_DEVICE=cuda:0 VEC=dummy NUM_ENVS=1 bash train.sh
```

Resume from latest run folder:

```bash
YOLO_DEVICE=cuda:0 VEC=dummy NUM_ENVS=1 bash train.sh --resume
```

Use a specific YOLO version/weights:

```bash
YOLO_VERSION=8 YOLO_WEIGHTS=./weights/yolo8n.pt bash train.sh
```

Torchvision detector example:
```bash
python train_single_stop_sign.py --detector torchvision --detector-model retinanet_resnet50_fpn_v2
```

Transformers DETR example:
```bash
python train_single_stop_sign.py --detector detr --detector-model facebook/detr-resnet-50
```

Evaluation (deterministic policy, logs to TensorBoard):

```bash
bash eval.sh
```

---

## Key Training Flags

From `train_single_stop_sign.py`:

- `--num-envs` (default 1 in `train.sh`) and `--vec` (`dummy` or `subproc`)
- `--n-steps`, `--batch-size`, `--total-steps` (PPO training control; `train.sh` defaults 1024/1024)
- `--episode-steps` (max steps per episode; default 300)
- `--grid-cell` (2, 4, 8, 16, 32) grid size in pixels (default 16)
- `--uv-threshold` UV drop threshold for success
- `--lambda-area` area penalty strength (encourages minimal patches)
- `--lambda-efficiency` efficiency bonus (drop per area)
- `--area-target` (default 0.25) target area fraction used for excess penalties
- `--step-cost` (default 0.012) and `--step-cost-after-target` (default 0.14) per-step penalties
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
- Phase penalties are uniform across phases (background/pole/transform are the only curriculum changes).
- `--bg-mode` (`dataset` or `solid`) and `--no-pole` for single-phase
- `--obs-size`, `--obs-margin`, `--obs-include-mask` (cropped observation + mask channel)
- `--ent-coef`, `--ent-coef-start`, `--ent-coef-end`, `--ent-coef-steps` (entropy coefficient schedule; default 0.001)
- `--detector-device` (e.g., `cpu`, `cuda`, or `auto`)
- `--detector` (`yolo`, `torchvision`, or `detr`) and `--detector-model` (model name for torchvision/DETR)
- `--step-log-every`, `--step-log-keep`, `--step-log-500` (step logging control)
- `--cnn` (`custom` or `nature`) choose feature extractor
- `--ckpt`, `--overlays`, `--tb` output paths (TB logs grouped under `grid_uv_yolo<ver>`)
- `--save-freq-steps` or `--save-freq-updates` checkpoint cadence
- `--check-env` runs SB3 env checker before training (enabled by default in `train.sh`)

---

## Environment Details

The environment is implemented in `envs/stop_sign_grid_env.py`.

Highlights:
- Discrete action space over valid grid cells inside the sign octagon.
- Action masking prevents duplicate cell selections (MaskablePPO).
- UV-on reward uses raw UV drop (`drop_on`) computed as the day baseline
  confidence minus UV-on overlay confidence.
- Reward includes an efficiency bonus (drop per area) plus fixed area penalties
  that push toward a target patch fraction.
- Optional per-step penalties can apply globally or only after the area target.
- Observations are cropped around the sign with an optional overlay-mask channel
  (controlled by `--obs-*` flags).
- Training uses a lightweight custom CNN extractor tuned for sign crops (or NatureCNN via `--cnn nature`).
- Area cap supports soft (penalty) or hard (terminate) modes.
- Minimum UV alpha (`uv_min_alpha`) ensures patches are visible under UV even with
  very low paint alpha.
- VecNormalize is applied to observations; evaluation should reuse the saved stats.

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

Efficiency bonus:
```
eff = log1p(max(0, drop_on) / max(area, efficiency_eps))
```

Core reward:
```
drop_cap = max(0, c0_day - success_conf)
drop_on = min(drop_on, drop_cap)
pen_day  = max(0, drop_day - day_tolerance)
raw_core = drop_on
         - lambda_day * pen_day
         - lambda_area * area
         - excess_penalty
         - step_cost_penalty
         + lambda_iou * (1 - mean_iou)
         + lambda_misclass * misclass
         + lambda_efficiency * eff
         - lambda_perceptual * perceptual_delta
```

Shaping + success:
```
shaping       = 0.35 * tanh(3.0 * (success_conf - c_on))
success_bonus = 0.2 * (1 - area)^2 if c_on <= success_conf else 0
raw_total     = raw_core + shaping + success_bonus
```

Excess penalty (when `area > area_target`):
```
excess = area - area_target
excess_penalty = lambda_area * (4.5 * excess + excess^2)
```

Step cost (global + target-scaled):
```
step_cost_penalty = step_cost
if area > area_target:
  step_cost_penalty += step_cost_after_target * (1 + (area - area_target)/area_target)
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
# train.sh (defaults)
tensorboard --logdir _runs/tb --port 6006
# eval.sh (defaults)
tensorboard --logdir _runs/tb_eval --port 6006
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
- `episode/lambda_area_used_final`
- `episode/area_target_frac_final`
- `episode/area_reward_corr` (rolling correlation between area and reward)

Step metrics:
- Rolling window of per-step rows in
  `_runs/tb/<run_id>/grid_uv_yolo8/<phase>/tb_step_metrics/step_metrics.ndjson`
- 500-step snapshots in
  `_runs/tb/<run_id>/grid_uv_yolo8/<phase>/tb_step_metrics/step_metrics_500.ndjson`
- Step scalars include reward components and area weights.

---

## Output Artifacts

Generated files:
- `_runs/checkpoints/<run_id>/` PPO checkpoints (run id like `yolo8_1`, `yolo11_2`, ...).
- `_runs/overlays/<run_id>/` best overlays (PNG + JSON) and `traces.ndjson` if enabled.
- `_runs/tb/<run_id>/` TensorBoard event files (grouped under `grid_uv_yolo<ver>/<phase>`).
- `_runs/tb_eval/` evaluation logs (if you use `eval.sh`).

Overlay saver:
- `utils/save_callbacks.py` keeps the best N overlays and appends trace metadata.
- Current training config disables overlay saving by default (`max_saved=0`).
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
- `tools/area_sweep_analyze.py` summarizes sweep results and generates plots.
- `tools/replay_area_sweep.py` replays logged sweep cases and saves images.
- `tools/test_stop_sign_confidence.py` checks detector confidence on a single image.
- `tools/cleanup_runs.py` removes old run outputs (defaults to `_runs`).
- `tools/detector_server.py` runs a shared detector (YOLO/torchvision/DETR) for multi-process training.
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

For torchvision/DETR, pass `--detector` and `--detector-model` (no `--model` needed):
```bash
python tools/detector_server.py --detector detr --detector-model facebook/detr-resnet-50 --device cuda:0 --port 5009
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
|-- .github/
|-- .venv/
|-- baselines/
|-- data/
|   |-- stop_sign.png
|   |-- stop_sign_uv.png
|   |-- pole.png
|   |-- backgrounds/
|
|-- detectors/
|   |-- factory.py
|   |-- remote_detector.py
|   |-- torchvision_wrapper.py
|   |-- transformers_detr_wrapper.py
|   |-- yolo_wrapper.py
|
|-- envs/
|   |-- stop_sign_grid_env.py
|
|-- slides/
|
|-- tools/
|   |-- aggregate_baselines.py
|   |-- debug_grid_env.py
|   |-- area_sweep_debug.py
|   |-- area_sweep_analyze.py
|   |-- eval_policy.py
|   |-- replay_area_sweep.py
|   |-- test_stop_sign_confidence.py
|   |-- cleanup_runs.py
|   |-- detector_server.py
|   |-- parse_tb_events.py
|   |-- run_baselines_compare.sh
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
|-- _runs_remote/
|
|-- train_single_stop_sign.py
|-- train.sh
|-- eval.sh
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
