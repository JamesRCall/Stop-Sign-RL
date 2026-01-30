# Gradient Patch Baseline (YOLO)

This folder contains a **separate**, gradient-based patch optimization baseline.
It does **not** modify or depend on the PPO training code paths.

## What it does
- Differentiable rendering of a stop sign + pole on backgrounds
- EOT-style transforms (affine, perspective, blur, noise, color jitter)
- Gradient optimization to reduce YOLO stop-sign confidence
- Patch constrained to your **6 UV paint colors** (or just yellow)

## Quick start

```bash
python baselines/gradient_patch/optimize_patch.py \
  --weights ./weights/yolo8n.pt \
  --palette uv6 \
  --color-mode global \
  --area-target 0.25 \
  --steps 2000
```

Only yellow:

```bash
python baselines/gradient_patch/optimize_patch.py \
  --weights ./weights/yolo8n.pt \
  --palette yellow \
  --color-mode global \
  --area-target 0.25 \
  --steps 2000
```

Outputs are written under:

```
baselines/gradient_patch/_runs/<run_id>/
```

Key files:
- `sign_day.png` / `sign_on.png`
- `patch_mask.png`
- `overlay_day.png` / `overlay_on.png` (RGBA overlays for evaluation)
- `metrics.json`

## Evaluation (PPO detector path)

Use the evaluator to score a gradient patch with the **same detector path**
as PPO (Ultralytics predict + NMS via `DetectorWrapper`).

```bash
python baselines/gradient_patch/eval_patch.py \
  --run baselines/gradient_patch/_runs/<run_id> \
  --weights ./weights/yolo8n.pt \
  --eval-k 10 \
  --transform-strength 1.0 \
  --bg-mode dataset
```

## Notes
- This uses raw YOLO outputs (no NMS) to keep gradients.
- Palette constraint is enforced via a softmax mixture of your UV paints.
- For a *single* real-world paint, use `--color-mode global`.

## TensorBoard

TensorBoard logs are written to `<run_dir>/tb` by default.

```bash
tensorboard --logdir baselines/gradient_patch/_runs --port 6007
```
