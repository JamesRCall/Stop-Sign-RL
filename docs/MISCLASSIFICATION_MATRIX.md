# Joint cell-material misclassification matrix

This guide describes the reproducible Linux simulation driven by
[`run_all_misclassification_models.sh`](../tools/run_all_misclassification_models.sh).
It is a synthetic, non-empirical software experiment. It does not establish a
physical attack, detector transfer, novelty, or publication acceptance.

## What the matrix runs

The default configuration performs targeted `stop sign` -> `traffic light`
misclassification because both labels are present in the stock COCO taxonomies
used by every configured detector. Detector checkpoints remain frozen. For
each detector and training seed, the runner trains a separate MaskablePPO patch
policy; it does not fine-tune or retrain the detector.

The default detector matrix contains nine frozen models:

| Result ID | Backend/checkpoint |
|---|---|
| `yolov8n` | Ultralytics YOLOv8n, local `weights/yolov8n.pt` |
| `yolo11n` | Ultralytics YOLO11n, local `weights/yolo11n.pt` |
| `fasterrcnn_resnet50_fpn_v2` | Torchvision Faster R-CNN ResNet-50 FPN v2 |
| `fasterrcnn_resnet50_fpn` | Torchvision Faster R-CNN ResNet-50 FPN v1 |
| `retinanet_resnet50_fpn_v2` | Torchvision RetinaNet ResNet-50 FPN v2 |
| `retinanet_resnet50_fpn` | Torchvision RetinaNet ResNet-50 FPN v1 |
| `ssd300_vgg16` | Torchvision SSD300 VGG-16 |
| `fcos_resnet50_fpn` | Torchvision FCOS ResNet-50 FPN |
| `rtdetr_r50vd` | Hugging Face `PekingU/rtdetr_r50vd` |

Torchvision and RT-DETR may download pretrained weights on first use. Cache
them before an offline run. The matrix configuration is
[`configs/misclassification_models.json`](../configs/misclassification_models.json);
the resolved configuration and its SHA-256 are saved with every run.

For a sign grid with `N` eligible cells and a palette with `P` materials, the
joint action space has `N * P` tokens. A token selects one canonical cell and
one material. A cell cannot be selected again with another material, so every
prefix is inclusion-monotone in geometry and has exactly one recorded material
assignment per selected cell. The default fixed-palette simulation exposes six
synthetic dual-state paint descriptors with white, red, green, yellow, blue,
and orange active colors and a shared translucent gray day state.

Before the first model, the matrix runs the unit-test suite once unless
`RUN_TESTS=0`. It then executes these stages for each selected model:

1. backend-specific dependency and asset checks;
2. one independently seeded policy-training run per requested seed;
3. development-only prefix generation and deterministic pattern selection;
4. frozen-pattern evaluation on fresh RNG seeds without policy inference;
5. random, forward-greedy, binary GA, Gaussian ES, binary PSO-family proxy, and
   reference-package CMA-ES searches; and
6. per-model and matrix CSV/JSON summaries, checksums, logs, provenance, and
   one top-level compressed archive.

The default five seeds and nine detectors therefore train 45 separate patch
policies. They also run six online baseline optimizers for every detector/seed
pair. These runs are sequential; set `MODEL_IDS` to split disjoint detector
subsets across machines or GPUs, and give each worker a different `OUTPUT`.

For the default targeted objective, a successful eligible transform requires
`traffic light` to be the highest-confidence detection overlapping the known
sign ROI, target confidence of at least 0.40, and localized `stop sign`
confidence of at most 0.20. The terminal joint predicate also enforces the
configured clean-eligibility, inactive/day-preservation, EOT-rate, and exact
area conditions. Report those components separately; a confidence decrease
alone is not misclassification.

## Training cap and stopping gate

The defaults are a hard maximum of 800,000 policy steps, a minimum of 50,000
steps, and an optional early stop when the terminal joint-success indicator is
at least 0.80 across a complete rolling window of 50 episodes. The hard maximum
always applies. A run that reaches it without satisfying the gate still
proceeds to frozen evaluation, where success may be lower or zero.

The gate is a compute-control rule, not convergence proof, model selection,
certification, or permission to discard unsuccessful seeds. Report whether the
gate fired, the final policy step count, every failed run, and every seed. Do
not increase the cap after inspecting only unfavorable methods unless that rule
was preregistered and applied symmetrically.

This remains a large experiment. With nine models, five seeds, four support
scenes, and training `K=2`, the hard-cap planning estimate is 576 million
overlay detector images, plus clean references at episode resets. Early
stopping can reduce that total but must not be assumed. The driver prints and
records the estimate; use the smoke run first and split disjoint `MODEL_IDS`
across machines when appropriate.

## Linux setup

From the repository root:

```bash
git pull --ff-only origin feature/gradient
git lfs pull

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt

test -f weights/yolov8n.pt
test -f weights/yolo11n.pt
test -f data/stop_sign.png
test -d data/backgrounds
python -m pytest -q tests
```

Use the CUDA-enabled PyTorch build appropriate for the host if the generic
dependency installation does not provide it. Confirm the selected device
before a long run:

```bash
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

## Smoke test

Run one detector and one short seed before launching the complete matrix:

```bash
export PYTHON_BIN="$PWD/.venv/bin/python"
export OUTPUT="$PWD/runs/misclassification_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
export DEVICE="cuda:0"
export MODEL_IDS="yolov8n"
export SEEDS="0"
export MAX_STEPS="4096"
export MINIMUM_STEPS="0"
export EARLY_STOP_SUCCESS_RATE="0"
export EVAL_EPISODES="10"
export QUERY_BUDGET="256"
export EVAL_K="2"
export RUN_TESTS="1"

bash tools/run_all_misclassification_models.sh
cat "$OUTPUT/STATUS.txt"
```

This is a plumbing check only. Its settings and results are not paper evidence.

## Full default run

Start a detached run whose verbose child output is written to files instead of
the VS Code terminal:

```bash
source .venv/bin/activate
export PYTHON_BIN="$PWD/.venv/bin/python"
export DEVICE="cuda:0"
export CONFIG="$PWD/configs/misclassification_models.json"
export MODEL_IDS=""
export SEEDS="0 1 2 3 4"
export MAX_STEPS="800000"
export MINIMUM_STEPS="50000"
export EARLY_STOP_SUCCESS_RATE="0.80"
export EARLY_STOP_WINDOW="50"
export EVAL_EPISODES="200"
export QUERY_BUDGET="10000"
export EVAL_K="8"
export TRAIN_EVAL_K="2"
export SUPPORT_SCENES="4"
export EPISODE_STEPS="64"
export MAX_PREFIX="64"
export RESUME="1"
export RUN_TESTS="1"
export CONTINUE_ON_ERROR="1"
export ARCHIVE="1"
export OUTPUT="$PWD/runs/misclassification_matrix_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTPUT"

nohup bash tools/run_all_misclassification_models.sh \
  >"$OUTPUT/launcher.log" 2>&1 &
echo $! | tee "$OUTPUT/launcher.pid"
```

The wrapper defaults supply the 800k/50k/0.80/50 training rule, 200 frozen
evaluation episodes, 10,000 detector-image queries per online baseline, `K=8`
paired EOT samples for development/evaluation, `K=2` during policy training,
four support scenes per training task, 64 environment steps, and a maximum
prefix length of 64.
To change them before launching, export `MAX_STEPS`, `MINIMUM_STEPS`,
`EARLY_STOP_SUCCESS_RATE`, `EARLY_STOP_WINDOW`, `EVAL_EPISODES`,
`QUERY_BUDGET`, `EVAL_K`, `TRAIN_EVAL_K`, `SUPPORT_SCENES`, `EPISODE_STEPS`,
or `MAX_PREFIX`.

Monitor without attaching training output to an interactive terminal:

```bash
cat "$OUTPUT/STATUS.txt"
tail -n 100 "$OUTPUT/launcher.log"
find "$OUTPUT/driver_logs" -type f -name '*.log' -exec tail -n 20 {} \;
watch -n 30 "cat '$OUTPUT/STATUS.txt'; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true"
```

Resume the same output directory after a host interruption:

```bash
export OUTPUT="/absolute/path/to/the/existing/matrix_directory"
export RESUME="1"
nohup bash tools/run_all_misclassification_models.sh \
  >>"$OUTPUT/launcher.log" 2>&1 &
echo $! | tee "$OUTPUT/launcher.pid"
```

Repeat the original run's exports when resuming; changing a budget or model
selection under an existing result root invalidates a clean comparison.
Completed stages are skipped. Incomplete training directories are moved to a
timestamped `.partial.*` sibling before that seed is restarted, preserving the
failed artifacts for diagnosis.

## Outputs to retain

The matrix root contains:

- `STATUS.txt`: overall `RUNNING`, `COMPLETED`, `FAILED_TESTS`, or `FAILED`
  state;
- `matrix_provenance.json`: arguments, selected models, Git commit, and config
  digest;
- `resolved_matrix_config.json`: absolute, validated inputs;
- `model_run_status.json`: per-detector exit status and log path;
- `driver_logs/<model>.log`: one complete child log per detector;
- `matrix_summary.csv` and `matrix_summary.json`: aggregated RL and baseline
  rows; and
- `misclassification_matrix_results.tar.gz`: the complete matrix archive.

Each `models/<model-id>/` directory contains its own `STATUS.txt`, generated
task/baseline configurations, per-stage logs and timing, `SHA256SUMS`,
and `simulation_summary.csv`. Child archives are disabled in matrix mode to
avoid duplicating every artifact inside the top-level archive. Under
`rl/seed_<n>/`, retain the completed run manifest, final policy,
VecNormalize state, generated prefix family, selected pattern, frozen
evaluation, and checkpoints. The run manifest's `training_stop` record states
whether the rolling gate or hard maximum ended training and seals policy-step
and offline detector-query accounting.

An empty summary is not success. Check both levels of `STATUS.txt`, every row in
`model_run_status.json`, and every detector log. The matrix continues to later
models by default after one child fails, but returns a failing exit status if
any selected model fails.

## Fair budgets and interpretation

Within one detector/seed task, native online baselines receive the same
detector-image limit, exact sign-alpha-pixel material cap, EOT seeds, objective,
and candidate contract. In joint-palette mode they search the same `N * P`
tokens, while a group constraint prevents selecting two colors for one cell.
All color tokens for a cell carry that cell's material-pixel cost; palette size
does not multiply the physical area budget.

The PPO row is not automatically equal-cost to an online baseline. Report its
offline policy-training detector queries separately from development and frozen
evaluation queries, and report deployment break-even cost as specified in
[`BUDGETED_COMPARISONS.md`](BUDGETED_COMPARISONS.md). The native
`fipatch_style_pso_proxy` is a PSO-family comparator, not a faithful FIPatch
reproduction. FIPatch, PatchAttack, and other named-paper claims require pinned,
audited adapters and declared candidate-space mappings.

## Speed-sign and physical-study boundary

Stock COCO models cannot distinguish `speed_limit_25` from `speed_limit_55`.
A fine-grained speed experiment requires licensed day/active assets and a
validated checkpoint for every selected backend whose exact label map contains
both source and target classes. Copy the matrix JSON, replace the attack labels
and assets, and replace the model list with those fine-grained checkpoints.
Do not relabel the default stop-sign matrix as speed-sign evidence.

The current joint-palette matrix uses fixed synthetic RGB material descriptors
and the same bundled sign image for its configured day and active assets. It
does not model measured fluorescence, fabrication, camera response, weather,
motion, or independent physical captures. Fresh frozen-evaluation RNG seeds do
not make the reused bundled sign/background collection a disjoint dataset. A
paper-facing physical study must replace these inputs with measured
spectral/camera records and disjoint trial splits, then use the certification
protocol rather than this debug matrix.

## Candidate contribution, not a novelty claim

The testable method hypothesis is a query-accounted black-box optimizer that
amortizes across declared tasks while emitting irreversible joint
geometry-material prefixes, with every prefix replayable as a frozen patch and
all matched baselines constrained to the same grouped action and material
budgets. That integrated formulation may be a useful contribution if experiments
show query/material gains and the required ablations isolate their cause.

It is not currently defensible to claim the first pixel-color patch, first RL
patch, first fluorescent patch, first misclassification patch, or first
speed-sign attack. Establishing a scoped novelty claim requires a systematic
literature review updated immediately before submission, faithful comparisons,
and empirical results. No script, result, or wording can guarantee novelty or
USENIX acceptance; see [`NOVELTY_AND_EVALUATION.md`](NOVELTY_AND_EVALUATION.md).
