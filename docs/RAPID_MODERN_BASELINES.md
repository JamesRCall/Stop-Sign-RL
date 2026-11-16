# Two new baselines on the existing five seeds

The runner executes **only FIPatch and IMPACT**, both from NeurIPS 2025.
DIRP, Greedy and Random are read from the existing saved comparison files;
they are never trained or rerun. All five table labels are plain method names.
The default new workload is 2 methods x 6 detectors x 5 seeds = 60 searches.

## Revision 3: sign geometry, material warm starts and meaningful refinement

The first server run completed but selected empty FIPatch masks in 20 of 30
searches. Circle centers are now repaired onto opaque sign support. Revision 3
also replaces the almost-fixed one-circle search with three variable-radius
circles, starts geometry search with the known UV material, and then allows
full RGB/geometry refinement. IMPACT repairs off-sign cells, mixes material
and unrestricted RGB starts, and refines tile content instead of adding tiny
independent noise to every high-resolution sign pixel.

The original 0.25 target was a soft reward preference, not a hard support
constraint. IMPACT now uses two native-DE populations with nominal 0.25 and
0.50 sign-area levels, splitting one total population of 16 between them.
They share the same total query ceiling. FIPatch's radius search spans small
patches through the maximum nominal support level. Actual clipped/overlapping
area is always measured. Larger patches are not treated as area-compliant:
ASR and the stricter joint success are reported separately.

The recommended rerun uses `suppression` (minimize activated confidence).
The backward-compatible default remains `dirp_score`. Changing the optimization
objective never changes the 0.20 success threshold, clean eligibility, or the
separate 0.30 area/daylight checks. Protocol settings are saved in every run.

Run a fixed pilot before scheduling another complete benchmark:

```bash
.venv-blackwell-baselines/bin/python -u tools/run_modern_baseline_pilot.py \
  --detector yolo8 --seed 1000 --queries 3078 \
  --output runs/modern_pilot_v3
```

This first checks that an empty overlay leaves images pixel-identical and that
the adapted renderer exactly reproduces the original grid renderer for a legal
yellow pattern. Full-sign black/yellow overlays are diagnostic controls only,
not benchmark attack successes. It then runs **both methods under both
objectives**, 512 candidates per search at K=3, with a one-hour cap per objective.
All four searches and their failures are retained. No saved baseline search is
repeated. A fresh output folder is required; old results are never replaced.

Read `pilot_summary.md`, `pilot_summary.json`, `renderer_checks.json`, and the
per-candidate traces. Check nonempty masks, confidence movement, clean-reference
differences, and completion. A functioning search can still fail to reach the
attack threshold; do not pick seeds, objectives, or outputs solely because they
produce favorable attack success. Confidence-only results are a different
optimization protocol and must not be pooled with the original-score results.

For a longer diagnostic search, predeclare a larger shared budget with
`--queries 12006` (2,000 candidates/search). Passing a renderer check does not
resolve the saved experiments' clean-reference mismatch or validate these
adaptations as original published-method reproductions. Preserve those limits
in any eventual report. Use a new output name for every changed protocol.

## Run on a Linux Blackwell server

Copy this repository, including `data/`, `weights/`, and
`runs/paper_data/compare/`, to the server. No DIRP policy checkpoint is needed.
Use Python 3.11 and a driver compatible with CUDA 12.8.

```bash
bash tools/setup_rapid_baselines_blackwell.sh
bash tools/run_rapid_baselines_v3_linux.sh
```

Setup/download time for the Python environment is separate from the timed
experiment. The experiment's 13.5-hour maximum includes fetching the pinned
optimizer source, detector model loading/downloads, calibration and searches.
The independent parent watchdog terminates a stalled GPU child. Export and
shutdown reserve leave the default command below 14 hours; it is a time cap,
not a guarantee that an unavailable model or slow server completes every run.

The recommended fixed-budget command uses **1,500 candidates / 9,006 detector
images per search**, including six clean-reference images at K=3. This is
90,000 candidates across 60 searches. It honors the requested common ceiling;
the wall watchdog can still stop incomplete searches. The optional `auto`
policy can reduce the ceiling during calibration to fit the time budget.
Partial searches are reported as
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
bash tools/run_rapid_baselines_v3_linux.sh --resume

# Refresh reports without GPU work or modifying source experiment files.
bash tools/run_rapid_baselines_linux.sh --report-only --output runs/modern_baselines_v3_rgb
```

Pass all original selection flags again when resuming a nondefault detector or
seed subset. A resume with changed input hashes, seeds, methods, budgets or
adapter code is rejected. Completed searches are skipped. Incomplete searches
start a new attempt; their old trace is retained separately.
Use a new output folder for revision 3; do not resume the earlier 0%-ASR run.

Optional material-matched comparison (a separate experiment):

```bash
bash tools/run_rapid_baselines_linux.sh \
  --color-mode material --search-objective suppression \
  --budget-policy fixed --queries 9006 --hours 13.5 \
  --output runs/modern_baselines_v3_material
```

Material mode fixes the original paint color; FIPatch spends its whole budget
on geometry PSO and IMPACT spends its whole budget on mask DE. Do not pool
material-mode and RGB-mode results or call the constrained variants native
paper reproductions. Do not launch both full experiments on the same GPU at once.

## What is preserved and what is adapted

Author implementations:

- [FIPatch](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html):
  pinned [author code](https://anonymous.4open.science/r/FIPatch-FA52/),
  its random-restart PSO wrapper and `pyswarm`. The adapter uses three circles,
  shared variable radius, swarm 16, 40 iterations per restart, and as many
  restarts as fit the external query budget. Material geometry receives 60%
  of queries; the remaining 40% jointly refines local geometry and full RGB.
  The residual RGB parameterization can reach every RGB color. Warm starts
  use the known paint, not successful masks from DIRP or the saved baselines.
- [IMPACT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html):
  pinned [author code](https://github.com/yangzh216/IMPACT) at commit
  `962cf92f26745e823a4abe3c664f9b414cafcce3`, including its population
  initialization, clustering/aggregation, mutation and crossover functions.
  Mask/content DE receives 70% of queries and fixed-mask tile-space (1+1)-ES
  receives 30%. ES uses coherent tile/block color changes (8--64 byte-scale
  Gaussian steps) and predefined palette probes, all counted in the budget.
  Off-sign repair preserves tile cardinality and color-to-tile association.

Hashes are checked before execution. The upstream experiments' top-level
dataset/model code is not executed. Only allowlisted optimizer functions are
loaded. Downloaded author files remain under the output directory for audit.

**These are detector/UV adaptations, not unmodified reproductions of the papers.**
The original loss functions and native image formation are replaced by the
common DIRP renderer and detector-confidence objective. IMPACT's original
4-pixel tiles become the benchmark's 16-pixel tiles. Population/iteration
counts are reduced for rapid evaluation and written to the run manifest.
There is a defensive guard for the author crossover's all-zero mutant case.
FIPatch activation alpha uses the common UV material rather than independently
fitting alpha. Its radius range is derived from actual opaque sign area and
the declared search-area levels. These initialization/geometry/refinement
changes must be described as part of the adaptation in any paper comparison.

In RGB mode the modern attacks optimize active RGB. DIRP/Greedy/Random retain the fixed
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

The legacy `dirp_score` includes stored area/daylight/efficiency penalties;
`suppression` searches on negative activated confidence alone. Neither changes
the recorded metrics or success gates. Actual support may exceed the original
soft area setting; inspect Joint ASR before claiming budget-compliant success.
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

- `comparison.md`, `comparison.csv`, `comparison.json`, `comparison.tex`: combined results and LaTeX table; CSV/Markdown distinguish confidence ASR from new-method Joint ASR.
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
