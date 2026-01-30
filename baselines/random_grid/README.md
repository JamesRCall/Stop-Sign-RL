# Random Grid Baseline

Randomly samples action sequences on the grid. Runs multiple trials and
keeps the best episode based on a selected score (default: `drop_on`).

## Run

```bash
python baselines/random_grid/random_search.py \
  --yolo-weights ./weights/yolo8n.pt \
  --eval-K 3 \
  --grid-cell 16 \
  --area-target 0.25 \
  --trials 50
```

Outputs go to:

```
baselines/random_grid/_runs/<run_id>/
```

Key files:
- `summary.json`
- `steps.json`
- `final_day.png`, `final_on.png`, `final_overlay.png`

