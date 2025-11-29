# UV-Adversarial Stop Sign Training Environment (YOLOv11 + PPO)

This project trains a PPO agent to **generate UV-reactive blob patterns** on a stop sign that remain
undetectable under daylight but cause YOLOv11’s stop-sign confidence to drop significantly when
the UV paint is *activated*.

The system now supports two-phase training (attack-only and UV-paint), realistic compositing with
poles and backgrounds, per-phase rewards, multi-color UV paints, and full TensorBoard + trace
logging.

> **Ethics Notice:** This project is solely for **academic adversarial robustness research**. Do not use
to cause harm, confusion, or safety risks.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Data Setup](#data-setup)
- [Training Phases](#training-phases)
- [Logging & Output](#logging--output)
- [Replaying Traces](#replaying-traces)
- [Customization](#customization)
- [License](#license)

---

## Project Overview

This environment simulates physical stop-sign attacks under two stages:

1. **Phase A – Attack-only:** The model learns blob patterns that lower YOLOv11 confidence
   when painted directly on the sign (no UV activation yet).
2. **Phase B – UV Paint Phase:** Uses realistic *UV paint pairs* (daylight vs UV-activated colors)
   to learn patterns that preserve high confidence before UV light but reduce it once activated.

Each training sample is rendered with:
- Realistic **pole + stop sign ratio**
- **Random backgrounds** (10–20 available)
- Controlled **camera transforms** (angle, blur, brightness, noise)
- UV paint color pairs defined in `utils/uv_paint.py`
- TensorBoard visual previews
- Per-step `trace` metadata for exact reproduction

---

## Directory Structure

```
.
├─ data/
│  ├─ stop_sign.png          # Base RGBA stop sign (transparent background)
│  ├─ stop_sign_uv.png       # Optional UV-reactive version of sign
│  ├─ pole.png               # Pole image (~81x960)
│  └─ backgrounds/           # 10–20 random driving scene backgrounds
│
├─ weights/
│  └─ yolo11n.pt             # YOLOv11 weights
│
├─ envs/
│  ├─ stop_sign_env.py       # PPO Gym environment (sign, pole, backgrounds, UV logic)
│  └─ random_blobs.py        # Blob geometry generator
│
├─ utils/
│  ├─ uv_paint.py            # Defines UV paint color pairs (day vs activated)
│  ├─ save_callbacks.py      # Top-50 saver + trace logging (replaces old saver)
│  ├─ tb_callbacks.py        # TensorBoard logging of scalars & overlays
│  └─ ...                    # Helper utilities
│
├─ runs/
│  ├─ checkpoints/           # PPO model saves
│  ├─ overlays/              # Top-50 best overlays (PNG + JSON)
│  ├─ overlays/traces.ndjson # Lightweight trace-only log
│  └─ tb/                    # TensorBoard event files
│
├─ train_single_stop_sign.py  # Main trainer with --mode attack/uv/both
├─ trace_replay.py            # Recreates blob masks or full composites from traces
└─ requirements.txt
```

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate      # or source .venv/bin/activate
pip install -r requirements.txt
```

If you’re on Linux or macOS, change the backslashes to `/`.

---

## Data Setup

Place the following in `/data`:

| File | Description |
|------|--------------|
| `stop_sign.png` | Transparent RGBA stop sign |
| `stop_sign_uv.png` | (Optional) Version photographed under UV light |
| `pole.png` | Narrow pole image (~81×960) |
| `backgrounds/` | Folder with 10–20 random backgrounds (640×640 recommended) |

---

## Training Phases

You can run each phase separately or both sequentially:

```bash
# Phase A: attack-only (no UV logic yet)
python train_single_stop_sign.py --mode attack

# Phase B: UV paint phase (requires defined UV paint pairs)
python train_single_stop_sign.py --mode uv

# Full curriculum (Phase A then Phase B)
python train_single_stop_sign.py --mode both
```

### Phase A
Learns blob geometry that reduces YOLO confidence directly.

### Phase B
Learns how to retain confidence in daylight (UV off) while lowering it under UV light.

---

## Logging & Output

### TensorBoard

```bash
tensorboard --logdir runs/tb --port 6006
```

Shows:
- **Scalars**: base_conf, after_conf, reward, step speed, etc.
- **Images**: recent composite overlays (sign + pole + background)

### Overlay Saver (`utils/save_callbacks.py`)
- Keeps **top 50** best overlays based on performance.
- Evicts the worst result once full.
- Writes `.png` (composite image) + `.json` (full info) + appends to `traces.ndjson` (compact trace-only row).

Example saved structure:
```
runs/overlays/
 ├─ adversary_step000123456_env00.png
 ├─ adversary_step000123456_env00.json
 ├─ traces.ndjson
```

Each JSON/trace row includes:
```json
{
  "phase": "B",
  "pattern_seed": 18239474,
  "transform_seed": 55612312,
  "place_seed": 88122,
  "count": 12,
  "size_scale": 1.1,
  "alpha_day": 0.4,
  "alpha_on": 0.8,
  "color_idx": 2
}
```

---

## Replaying Traces

Recreate the exact blob mask or composite from saved seeds:

```bash
# Mask only (just blob trace, pre-transform)
python trace_replay.py --just-mask

# Choose by index or step number
python trace_replay.py --index 12 --variant day
python trace_replay.py --step 250000 --variant on
```

Outputs:
```
replay_out/
 ├─ trace_mask_pre_B_step250000.png   # Blob mask (white blobs, black background)
 ├─ replay_B_day_sign_rgba.png        # Optional full sign variant
 ├─ replay_B_day_final.png            # Full scene composite (if not --just-mask)
```

---

## Customization

- **UV Paint Colors** → define multiple `UVPaint` pairs in `utils/uv_paint.py`, e.g.:
  ```python
  GREEN_GLOW = UVPaint("#5aff6e", "#cfff5a", translucent=True)
  BLUE_GLOW  = UVPaint("#0066ff", "#00ffff", translucent=True)
  ```
  and pass them in `train_single_stop_sign.py`:
  ```python
  UV_PAINTS = [VIOLET_GLOW, GREEN_GLOW, BLUE_GLOW]
  ```

- **Pole ratio & placement** handled automatically in `_compose_sign_and_pole()`  
  Adjust `pole_width_ratio` or `bottom_len_factor` in that function if needed.

- **Blob realism**: edit `_random_transform_sign()` for angle/blur/noise strength.

- **Trace limit**: change `max_saved` in `SaveImprovingOverlaysCallback` to control how many
  top results are retained.

---

## License

MIT — use for research and educational purposes only.
