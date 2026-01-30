# Greedy Grid Baseline

Greedy search over grid cells. At each step, it evaluates **all valid cells**
and selects the action that maximizes a chosen score (default: reward).

## Run

```bash
python baselines/greedy_grid/greedy_search.py \
  --yolo-weights ./weights/yolo8n.pt \
  --eval-K 3 \
  --grid-cell 16 \
  --area-target 0.25
```

Outputs go to:

```
baselines/greedy_grid/_runs/<run_id>/
```

Key files:
- `summary.json`
- `steps.json`
- `final_day.png`, `final_on.png`, `final_overlay.png`

