# Two new baselines on the existing five seeds

The runner executes **only FIPatch and IMPACT**, both from NeurIPS 2025.
DIRP, Greedy and Random are read from the existing saved comparison files;
they are never trained or rerun. All five table labels are plain method names.
The default new workload is 2 methods x 6 detectors x 5 seeds = 60 searches.

## Run on a Linux Blackwell server

Copy this repository, including `data/`, `weights/`, and
`runs/paper_data/compare/`, to the server. No DIRP policy checkpoint is needed.
Use Python 3.11 and a driver compatible with CUDA 12.8.

```bash
bash tools/setup_rapid_baselines_blackwell.sh
bash tools/run_rapid_baselines_linux.sh --output runs/modern_baselines_14h
```

Setup/download time for the Python environment is separate from the timed
experiment. The experiment's 13.5-hour maximum includes fetching the pinned
optimizer source, detector model loading/downloads, calibration and searches.
The independent parent watchdog terminates a stalled GPU child. Export and
shutdown reserve leave the default command below 14 hours; it is a time cap,
not a guarantee that an unavailable model or slow server completes every run.

Calibration measures the selected detectors and chooses **one common query
ceiling** for both new methods before any searches run. It can reduce the
requested ceiling to fit the time budget. Partial searches are reported as
missing, never as successful completed tests or ordinary attack failures.
Exit code 0 means every planned new search completed, 2 means partial results,
1 means an error, and 130 means an interrupted run. Review `run_status.json`.

The default seeds are the original **1000,1001,1002,1003,1004**. Examples:

```bash
# One original seed, all six detectors; saved old methods are subset to that seed too.
bash tools/run_rapid_baselines_linux.sh --seeds 1000 --output runs/modern_seed1000

# One detector, original five seeds.
bash tools/run_rapid_baselines_linux.sh --detectors yolo8 --output runs/modern_yolo8

# Resume the same frozen plan/inputs. Each invocation has its own 13.5-hour cap.
bash tools/run_rapid_baselines_linux.sh --resume --output runs/modern_baselines_14h

# Refresh reports without GPU work or modifying source experiment files.
bash tools/run_rapid_baselines_linux.sh --report-only --output runs/modern_baselines_14h
```

Pass all original selection flags again when resuming a nondefault detector or
seed subset. A resume with changed input hashes, seeds, methods, budgets or
adapter code is rejected. Completed searches are skipped. Incomplete searches
start a new attempt; their old trace is retained separately.

## What is preserved and what is adapted

Author implementations:

- [FIPatch](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html):
  pinned [author code](https://anonymous.4open.science/r/FIPatch-FA52/),
  its random-restart PSO wrapper and `pyswarm`, circular support and optimized
  per-circle RGB and a shared radius. Defaults follow the author code: one
  circle, radius 7/32 of sign size, swarm size ten, 30 iterations and five
  random restarts, truncated only by the external query/time ceiling.
- [IMPACT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html):
  pinned [author code](https://github.com/yangzh216/IMPACT) at commit
  `962cf92f26745e823a4abe3c664f9b414cafcce3`, including its population
  initialization, clustering/aggregation, mutation and crossover functions.
  Mask/content DE is followed by fixed-mask pixel-level Gaussian (1+1)-ES.

Hashes are checked before execution. The upstream experiments' top-level
dataset/model code is not executed. Only allowlisted optimizer functions are
loaded. Downloaded author files remain under the output directory for audit.

**These are detector/UV adaptations, not unmodified reproductions of the papers.**
Both papers' classifier losses and native image formation are replaced by the
common DIRP renderer and detector-confidence objective. IMPACT's original
4-pixel tiles become the benchmark's 16-pixel tiles. Population/iteration
counts are reduced for rapid evaluation and written to the run manifest.
There is a defensive guard for the author crossover's all-zero mutant case.
FIPatch activation alpha uses the common UV material rather than independently
fitting alpha, and its circle radius is scaled from the author's 32px input to
the benchmark sign size.

The modern attacks optimize active RGB. DIRP/Greedy/Random retain the fixed
yellow material from the saved experiments. Inactive appearance and activation
alpha come from the benchmark UV-paint model; arbitrary active RGB is a
digital modeling freedom, not proof that every optimized color is physically
realizable in the available ink. Report this adaptation and color freedom in
the methods/caption if using the comparison in the paper. No physical attacks
or new physical measurements are performed by this code.

## Seed and setting checks

Each detector's environment configuration comes from its original five-episode
`compare_summary.json`. Seed IDs, `K=3`, sign assets, background directory,
grid-cell size, area target, and original soft area-cap settings are retained.
The old 0.30 area setting was a **soft penalty**, not a hard limit. The runner
does not impose a new hard 0.30 constraint on only the new methods.

Score = suppression plus the stored area/daylight/efficiency penalties;
localization/misclassification reward terms are not used by the new score-only
adapters. ASR@0.2 uses the mean image-wide maximum stop-sign confidence over
the three matched transformed views. Clean eligibility, daylight preservation,
area-cap exceedance and the stricter joint success are recorded separately.
The selected patch prefers successful candidates and then lower area, matching
the saved Random winner convention. Every candidate is retained in a trace.

Matching seed IDs alone does **not** prove the old scenes were reconstructed:
old background ordering, assets and renderer/library versions can differ.
Every run saves transformation/place seeds, image hashes, the current clean
confidence, and its difference from the original recorded clean confidence.
The report explicitly flags differences above 0.001. Resolve such differences
before describing the comparison as exact paired-scene reproduction. The
matched old renderer/data snapshot is needed if exact old scene replication
is required; the five saved summary files cannot recover missing old pixels.

Each detector image counts as one query, including the 2*K clean references.
Batching does not reduce the counted queries. The shared query ceiling applies
to the two new methods. Old methods are reused at their recorded budgets;
this is not retroactively an equal-query comparison across all five methods.
The output records the saved/new origin in provenance without changing the
five method names in the table.

## Results and progress

Console progress includes candidates, detector-image queries, trial ETA and
remaining wall time. `progress.json` is updated during each search.

- `comparison.md`, `comparison.csv`, `comparison.json`, `comparison.tex`: combined five-method results, including a paper-ready LaTeX table.
- `per_seed.csv`: per-seed values and completion/source/reference-check fields.
- `manifest.json`, `plan.json`: input/source hashes, fixed budget, timing plan.
- `environment/*.json`: GPU, package versions and exact per-detector configuration.
- `<detector>/<seed>/<method>/result.json`: status, best candidate and measured cost.
- `attempts/*/trace.jsonl`: every scored proposal and its query count.
- `best_patch.npz`, `best_overlay.png`, `best_day.png`, `best_on.png`: replayable patch and previews.

Random aggregates **all 50 recorded trials per saved run**, not only the
best-of-50 candidate. Five seeds therefore retain 250 trial records with 54
distinct identifiers. Random runtime is per candidate. Old timing values are
not rerun on Blackwell: do not claim hardware-normalized speedups from those
columns. Saved JSON values are used directly, including the corrected
RetinaNet DIRP mean runtime, rather than the stale 0.328-minute table entry.

For Blackwell installation details, see the official
[PyTorch 2.7 release](https://pytorch.org/blog/pytorch-2-7/) and
[CUDA 12.8 wheel instructions](https://pytorch.org/get-started/previous-versions/).
