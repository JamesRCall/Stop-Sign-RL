# Budget-matched black-box comparisons

This document defines the comparison contract implemented in
`baselines/budgeted/`. It is an experiment harness, not a bundle of empirical
results and not evidence that a named method has been faithfully reproduced.
Run it only with owned or expressly authorized sign replicas in a controlled
setting.

The synthetic nine-detector driver applies this contract automatically; see
the [misclassification matrix guide](MISCLASSIFICATION_MATRIX.md). That runner
is a software simulation and does not make its outputs physical evidence.

## What is matched

Every admitted run receives a fresh oracle with the same scene, sign, detector,
class objective, fixed material or ordered material palette, action encoding,
grid, transformation seeds, thresholds, and scalar reward. The suite hashes
that contract and rejects a method if its fingerprint differs from the first
method's fingerprint.

Two candidate spaces are supported. Fixed-material mode has one binary token
per eligible canonical cell. Joint-palette mode has one token per
cell/material pair. All tokens belonging to one cell share a group, and the
evaluator rejects a candidate that selects more than one token from that group.
This gives every optimizer the same color choices without allowing it to spend
one cell's area multiple times or assign contradictory materials.

The hard budgets are:

- **Detector images.** One image passed to one detector counts as one query,
  even when inference is batched. The total includes the `2 * K` clean day and
  activated reference images. Each candidate costs another `2 * K` images.
  Fixed EOT is mandatory; adaptive `K` is rejected.
- **Material pixels.** A cell costs the exact number of source-sign alpha-mask
  pixels that it covers. Every material token for that cell has the same cost;
  palette size does not multiply the area budget. A candidate that selects two
  materials for one cell or exceeds the inclusive integer limit is rejected
  before detector inference. This image-space measure is not physical ink
  volume; report measured mass or volume separately in physical experiments.

The oracle reports the same localized disappearance, untargeted
misclassification, or targeted-misclassification objective used by the Gym
environment. The top-level `joint_success` field is the complete predicate:
clean eligibility, the configured attack condition, inactive/day preservation
when required, and the environment area constraint. Nested
`objective_condition_success` isolates only the selected disappearance or
misclassification condition. The trace retains every tested candidate, so
success-versus-query and success-versus-area curves must be computed from the
full trace rather than only the scalar-score winner.

## Implemented method slots

| Method ID | Execution | Claim boundary |
|---|---|---|
| `random_search` | Native | Samples physical-cell inclusion rates, then one uniformly chosen material token per included cell, followed by exact material repair; not uniform over all feasible assignments. |
| `forward_greedy` | Native | Seeded, query-bounded forward marginal addition. |
| `genetic_algorithm` | Native | Binary-token GA with uniform crossover, mutation, and exact group/material repair. |
| `gaussian_es` | Native | Isotropic Gaussian logit ES decoded to a grouped feasible assignment; it is not CMA-ES or NES. |
| `cma_es` | Reference package | The installed `cma` implementation optimizes continuous priorities which are deterministically decoded to grouped feasible assignments. |
| `fipatch_style_pso_proxy` | Native proxy | Executable binary-token PSO-family comparator with grouped repair; never report it as FIPatch. |
| `fipatch_pso` | External adapter | Requires pinned upstream code and an audited mapping from FIPatch's native parameters into the common candidate/material contract. |
| `patchattack` | External adapter | Requires pinned upstream code, texture assets, and an audited detector/grid adaptation. The original PatchAttack is an ImageNet texture/position attack, so a grid adaptation is not the unmodified ECCV method. |
| `baap_2606_18318` | External adapter | Requires a pinned implementation and an audited mapping for its location, texture, and size variables. |
| `per_task_ppo` | External adapter | Requires one frozen, independently trained policy artifact per task. |
| `wei_rl_2212_12995` | External adapter | Requires a pinned implementation and audited traffic-sign adaptation. |
| `meta_attack_iccv21` | External adapter | Requires a pinned implementation and an audited support/query-task adaptation. |
| `impact_de_es` | External adapter | Requires a pinned implementation and an audited irregular-mask/material mapping. |
| `simulator_attack_cvpr21` | External adapter | Tests the learned-prior/query-amortization hypothesis; requires a patch/detector adaptation and separate offline simulator-query accounting. |

The machine-readable registry is emitted with every report. A named external
slot fails before candidate queries unless its manifest, revision, artifact
hash, runner identity, and restricted query interface agree. Those integrity
checks establish which adapter ran; they do not by themselves prove semantic
equivalence to a paper. PatchAttack's authors provide their original
[ECCV 2020 code](https://github.com/Chenglin-Yang/PatchAttack), but this
repository does not silently coerce that classifier-oriented implementation
into a traffic-sign detector result.

The registry boundaries follow the primary descriptions of
[FIPatch](https://papers.nips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html),
[PatchAttack](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5563_ECCV_2020_paper.php),
[Wei et al.](https://pubmed.ncbi.nlm.nih.gov/37015667/),
[Meta-Attack](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Meta-Attack_Class-Agnostic_and_Model-Agnostic_Physical_Adversarial_Attack_ICCV_2021_paper.html),
[Simulator Attack](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.html),
[IMPACT](https://papers.nips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html),
and the June 2026
[budget-aware method](https://arxiv.org/abs/2606.18318).

## Native command

Create an environment JSON using the same fields as the existing baseline
scripts. The non-runnable
[`budgeted_comparison.template.json`](../configs/budgeted_comparison.template.json)
and [`budgeted_methods.template.json`](../configs/budgeted_methods.template.json)
files are starting points. For a targeted fine-grained speed-sign study, a
minimal study-specific file resembles:

```json
{
  "data": "./data",
  "sign_profile": "custom",
  "sign_image": "./data/speed_25_day.png",
  "sign_active_image": "./data/speed_25_active.png",
  "source_class": "speed_limit_25",
  "attack_mode": "targeted_misclassification",
  "attack_target_class": "speed_limit_55",
  "yolo_weights": "./weights/fine_grained_traffic_sign.pt",
  "bgdir": "./data/backgrounds_development",
  "paint_action_mode": "joint_palette",
  "paint_palette": "white,red,green,yellow,blue,orange",
  "action_indexing": "canonical_full_grid",
  "eval_K": 8,
  "area_cap_frac": 0.2,
  "transform_strength": 1.0
}
```

The assets and fine-grained checkpoint above are placeholders and are not
shipped by this repository. Then run:

```powershell
python tools/run_budgeted_comparison.py `
  --environment-json ./configs/comparison_speed_25_to_55.json `
  --methods random_search,forward_greedy,genetic_algorithm,gaussian_es,fipatch_style_pso_proxy,cma_es `
  --detector-query-limit 10000 `
  --material-area-fraction 0.20 `
  --scene-seed 1001 `
  --optimizer-seed 7 `
  --output ./runs/comparisons/speed_25_to_55_scene_1001_seed_7.json
```

An optional method-configuration file maps method IDs to keyword arguments:

```json
{
  "genetic_algorithm": {"population_size": 32},
  "gaussian_es": {"population_size": 32, "sigma": 1.0},
  "fipatch_style_pso_proxy": {"swarm_size": 32},
  "cma_es": {"population_size": 32, "sigma": 1.0}
}
```

Pass it with `--method-config-json`. Do not give methods unequal
`max_evaluations` values in a matched-budget experiment. The hard detector
guard remains authoritative even when a method asks for more evaluations.
Reports are written atomically and existing paths are not overwritten.

## Named external methods

`run_external_adapter` in `baselines/budgeted/external.py` is the integration
boundary for a reviewed wrapper. The wrapper may initialize its internal state
and propose a canonical cell mask from the complete accounted history; it may
not call the detector directly. The adapter verifies a manifest containing the
method ID, paper URL, upstream repository and revision, wrapper artifact path
and SHA-256, and runner API version. The current command-line tool intentionally
runs only native slots; invoke reviewed external adapters from a study-specific
driver and release that driver, manifest, dependency lock, mapping rationale,
and hashes with the artifact.

The deliberately invalid
[`external_baseline_manifest.template.json`](../configs/external_baseline_manifest.template.json)
and
[`external_candidate_mapping.template.json`](../configs/external_candidate_mapping.template.json)
show the required records. Replace their placeholder revisions, hashes, audit
status, adaptation differences, and offline query count; do not make the status
valid before the mapping has actually been preregistered and independently
reviewed.

The common oracle accepts either fixed-material cell masks or grouped
cell-by-material tokens. A method whose native variables include continuous
texture, free patch position, shape, blending, or a different material model
still needs a preregistered mapping. A joint-palette adapter must declare the
canonical token encoding and may never select two material tokens for one cell.
Report such a run as an adaptation unless the authors' original algorithm and
search space are genuinely preserved. A pinned wrapper is necessary but not
sufficient for a fidelity claim.

## Paper experiment matrix

One CLI invocation covers one fixed scene/task and one optimizer seed. It is not
a paper-level comparison. Before inspecting results, preregister a matrix over:

- disappearance, untargeted, and targeted objectives;
- stop signs and fine-grained speed-sign source/target pairs;
- disjoint sign instances, backgrounds, detectors, cameras, and material
  batches;
- at least five independent optimizer/training seeds; and
- several common query and material budgets sufficient to recover a Pareto
  curve.

For joint-palette experiments, freeze palette membership and order before
opening outcomes. Run geometry-only fixed-material controls for every palette
entry and report the larger `N * P` search space. Do not give the proposed
policy access to color while constraining comparators to geometry alone.

Aggregate macro and per-task joint ASR with uncertainty intervals. Also report
clean eligibility, inactive preservation, exact material pixels and fraction,
wall-clock time, failures, and unavailable baseline slots. Do not drop a method
because it exhausted its budget or failed on a hard task.

For the amortized RL method, separate offline policy-training detector images
from online target-task images. Compare both equal-online-query performance and
total cost after `D` deployments:

```text
amortized total(D) = offline training queries + D * online queries per task
per-task total(D)  = D * per-task search queries
```

Report the observed break-even deployment count rather than treating offline
queries as free. Match wall-clock and measured material budgets as secondary
analyses; equal detector-image queries alone do not make compute or fabrication
costs equal.

## Interpretation limits

The harness verifies accounting and configuration invariants, not empirical
superiority, physical transfer, sample independence, or novelty. It does not
yet aggregate a full task manifest, execute external named-paper adapters from
the CLI, measure ink volume, or ship paper-level comparison results. Grouped
cell-material accounting establishes a fair discrete contract; it does not
show that the palette is physically realizable or that the enlarged action
space is novel. Until the required studies exist, the defensible statement is
“the repository implements a matched-budget evaluation contract,” not “the
proposed method outperforms FIPatch, PatchAttack, or other prior work.”
