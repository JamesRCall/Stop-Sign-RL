# Dual-State Traffic-Sign Patch Optimization

This repository studies sequential, black-box optimization of sparse patches whose
inactive/day state should preserve traffic-sign perception while an activated
state causes a configured detector error. It contains a legacy single-task path
and an experimental v2 path for task-amortized, target-conditioned optimization.
Both use action-masked PPO and the same localized paired-state objective boundary.

> Research status: this is a refactored research prototype, not a completed
> artifact and not evidence of guaranteed publication. No code change can
> guarantee USENIX acceptance. Results produced before the localized,
> mode-specific objective refactor are stale and must be regenerated. No
> empirical v2, speed-sign, fluorescence-fidelity, or certification result is
> included.

The closest prior work is
[FIPatch (*The Fluorescent Veil*, NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html),
which already covers fluorescent/UV traffic-sign patches, EOT, black-box
optimization, area reduction, stop and speed signs, hiding, and misrecognition.
Do not claim novelty for that premise. See
[the novelty and evaluation protocol](docs/NOVELTY_AND_EVALUATION.md) before
writing the paper or running headline experiments.

All physical work must use owned replicas or expressly authorized signs in a
controlled area. Never modify deployed public traffic-control infrastructure.

## Experimental v2 research claim

The exact proposed claim sentence is deliberately a question, not a result:

> We study whether a target-conditioned optimizer trained across a
> preregistered task distribution can reduce online query and material costs by
> emitting inclusion-monotone fluorescent-stencil prefixes, with the smallest
> qualifying prefix selected on calibration trials and evaluated on disjoint
> certification trials under measured spectral/camera inputs and simultaneous
> finite-sample risk bounds.

The repository implements the contracts needed to test that sentence. It does
not yet establish an online-query reduction, a material reduction, physical
fidelity, generalization, or novelty relative to all concurrent work. See the
[method contract](docs/PREFIX_VALID_AMORTIZED_METHOD.md) and
[novelty/evaluation protocol](docs/NOVELTY_AND_EVALUATION.md) before framing a
paper claim.

The intended v2 data flow is:

~~~text
strict task manifest + measured calibration
                  |
                  v
train-only target-conditioned support-batch policy
                  |
                  v
development-only hashed inclusion-monotone prefix family
                  |
                  | strict hash/task/area/query binding + trial inventory
                  v
preregistered calibration selection -> sealed task-indexed order -> disjoint certification
~~~

The stages are intentionally separable and auditable; the repository does not
yet ship a one-command detector/capture experiment or paper-ready result bundle.
The risk plan still requires a researcher-supplied trial inventory that must be
externally registered before outcomes are opened, and detector or physical-
capture outcomes still require explicit result rows. Task-manifest
`calibration`/`certification` entries are not silently converted into results.

## What is implemented

- Alpha-mask geometry supports octagonal, circular, rectangular, and custom sign
  assets without hard-coded stop-sign cell coordinates.
- Explicit source and designated attack-target labels resolve against the
  detector label map and fail closed when absent.
- Three objectives share one implementation in **envs/attack_objective.py**:
  source-class evasion (legacy CLI name “disappearance”), localized untargeted
  misclassification, and localized targeted misclassification.
- Wrong-label detections count only when their boxes overlap the known rendered
  sign ROI. An unrelated object elsewhere in the image cannot produce success.
- Untargeted experiments can preregister a traffic-sign label set with
  **--allowed-alternative-classes**.
- Day and activated views use matched backgrounds, placements, and transforms.
  Active suppression is measured as c0_on − c_on, not against a mismatched
  daylight asset.
- Joint success enforces clean baseline eligibility, the selected EOT objective,
  inactive/day preservation, and the exact painted-pixel area cap.
- Detector failures raise errors rather than becoming false disappearance
  successes.
- Detector-image queries, seeds, asset/weight hashes, package versions, and
  checkpoint-specific VecNormalize state are recorded.
- **tools/eval_frozen_pattern.py** evaluates one immutable stencil on fresh
  certification RNG seeds without policy inference or search.
- A strict [task-manifest loader](utils/task_manifest.py) defines train,
  development, calibration, and certification tasks; hashes source and canonical
  content; resolves paths; and rejects configured leakage across sign instances,
  background collections, cameras, material batches, physical-run groups, and
  source-target pairs. The bundled
  [manifest template](configs/amortized_tasks.template.json) is intentionally
  non-runnable.
- [train_amortized.py](train_amortized.py) samples only manifest tasks labeled
  `train`, requires at least two training tasks and all held-out splits in its
  paper-facing mode, conditions the policy on source/target, detector,
  calibration, constraints, and declared task features. A completed run seals
  offline detector-query totals plus final policy and VecNormalize hashes in
  `amortized_run_manifest.json`.
- [envs/amortized_traffic_sign_env.py](envs/amortized_traffic_sign_env.py)
  applies each canonical cell action to every support scene, rejects repeated or
  invalid actions, records the ordered prefix and its hash, aggregates a lower-
  tail empirical CVaR objective, and exposes explicit constraint violations and
  dual variables. That prefix hash covers grid shape and ordered cell indices,
  not a complete physical artifact.
- [tools/generate_amortized_prefixes.py](tools/generate_amortized_prefixes.py)
  loads the frozen policy, normalization state, and completed run manifest;
  generates a capped ordered sequence on development tasks only; records exact
  painted image-pixel area and query counts; binds task, sign-asset,
  material/calibration, policy, and normalizer hashes; and emits a
  content-addressed candidate-family artifact. Generation stops at declared
  prefix/query limits or environment termination, and the shared family ends at
  the shortest task sequence. It does not read calibration/certification
  outcomes or issue a certificate.
- [tools/build_risk_protocol.py](tools/build_risk_protocol.py) verifies the
  candidate-family self-hash, rejects debug artifacts, requires exact agreement
  between candidate and researcher-supplied inventory task IDs, and binds
  pattern hashes, exact area, task IDs, and pre-evaluation query totals into the
  strict risk protocol. The resulting protocol carries the candidate-family
  SHA-256. That self-hash establishes internal consistency, not timing or
  authorship. Externally register the built protocol together with the family
  artifact/digest and inventory before outcomes are observed.
- The [spectral transport module](utils/fluorescence_transport.py) and
  [v1 schema](schemas/fluorescence_transport_v1.schema.json) require wavelength,
  illuminant/UV, substrate, material, camera/ISP, provenance, and bounded
  uncertainty inputs. The included
  [synthetic fixture](data/synthetic/fluorescence_transport_v1.synthetic.json)
  is rejected by paper-facing training unless an explicit debug override is
  used; it is not measurement evidence. Training currently converts each seeded
  transport result to one opaque effective day/active sRGB cell color per
  support replica rather than performing spectral transport inside the renderer.
- [tools/certify_attack_results.py](tools/certify_attack_results.py) implements
  separate calibration selection and final certification over supplied result
  rows, with fixed sample lists, disjoint declared sample hashes, exact one-sided
  binomial bounds, simultaneous error control, exact material-area checks, and
  fail-closed query reconciliation. It validates records; it does not run the
  detector or prove that declared samples are physically independent. See the
  [certification protocol](docs/CERTIFICATION_PROTOCOL.md).
- [baselines/budgeted](baselines/budgeted) provides one fail-closed query and
  exact image-pixel material ledger for native random, forward-greedy, binary
  GA, Gaussian ES, binary PSO-family, and reference-package CMA-ES runs. Named
  FIPatch, PatchAttack, budget-adaptive, per-task RL, Meta-Attack, Simulator
  Attack, Wei et al., and IMPACT slots remain unavailable until a pinned runner
  and declared, preregistered candidate-space mapping are supplied; an
  executable proxy is never relabeled as the paper method. See the
  [budgeted-comparison protocol](docs/BUDGETED_COMPARISONS.md).
- Unit tests cover objective boundaries, ROI attribution, target requirements,
  speed-sign geometry, exact area, deterministic reset, detector failures,
  strict class lookup, task leakage, prefix invariants, spectral transport,
  risk bounds, and frozen evaluation.

## Important scope limits

Speed-sign support is infrastructure, not a completed speed-sign result. The
repository does not ship a validated fine-grained traffic-sign detector or a
licensed speed-sign asset suite. Standard COCO checkpoints expose “stop sign”
but do not distinguish speed-limit values such as 25 and 55. A defensible 25→55
study requires custom fine-grained weights, a complete class map, licensed
assets, clean-accuracy validation, disjoint splits, and physical experiments.

The legacy `train_traffic_sign.py` PPO path constructs a new stencil for each
scene and remains a scene-conditioned optimizer. The v2 trainer instead learns
one policy across manifest tasks and holds one stencil fixed across a task's
support scenes. A frozen v2 policy may still emit a different sequence for each
new task; that is an amortized optimizer, not a universal paint-once patch. A
universal claim requires fixing one stencil before held-out evaluation.

The v2 implementation has unit-tested software invariants, not trained-policy or
physical evidence. Its task template references placeholder sign assets,
detectors, and synthetic calibration. Replace all placeholders with licensed
assets, validated fine-grained weights, and measured calibration before a
paper-facing run. Debug overrides such as `--allow-uncalibrated-simulation`,
`--allow-single-task-debug`, and `--allow-incomplete-splits` invalidate the
corresponding physical, amortization, or held-out claims.

V2 conditioning is not yet a demonstrated compositional task representation.
Several identities enter as short deterministic hash fingerprints; declared
condition-feature names and lengths are checked, but their semantics are not.
Background and physical-run IDs are not direct policy-vector fields, and the
image observation is the first support replica while the others affect aggregate
feedback and reward. Treat cross-task generalization as an empirical question.

Likewise, “prefix-valid” currently means digitally inclusion-monotone canonical
cell indices with exact rendered-image-pixel area. The development pattern
descriptor does not fully encode fabrication dimensions or volume, detector and
camera identity, the applied paint descriptor, or the spectral transport
draw/seed. Do not call a generated prefix physically fabricable until a complete
physical artifact is exported, hashed, fabricated, and checked.

The implemented “disappearance” predicate is source-class evasion: localized
source confidence is below its threshold. It does not prove that all
traffic-sign objectness disappeared. Use that precise term in the paper unless
a detector-wide traffic-sign/objectness predicate is added.

## Attack definitions

For source class s, optional target t, known sign box B, localization threshold
eta, source threshold tau_s, and target threshold tau_t:

- A transform is clean-eligible only when both matched clean day and clean
  activated views localize s at B above the baseline threshold.
- Source-class evasion requires localized p(s) ≤ tau_s.
- Untargeted misclassification requires a localized allowed alternative label
  with confidence at least tau_t, winning over s, and—by default—source
  suppression. If no allowlist is supplied, every non-source detector label is
  eligible; only use that setting when the complete detector taxonomy consists
  of traffic-sign classes.
- Targeted misclassification requires t to be the highest-confidence localized
  label, p(t) ≥ tau_t, and—by default—source suppression.
- The objective must hold for **--min-attack-success-rate** of clean-eligible EOT
  samples.
- Joint success additionally requires the configured clean-eligibility rate,
  day correctness and confidence-drop tolerance, and exact area budget.

Report clean eligibility, objective-only rate, inactive preservation, joint
ASR, exact painted area, source/target confidence and IoU, and detector queries
separately. Do not relabel confidence suppression as misclassification.

## Environment setup

Python 3.10 is the reference version.

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

A Conda specification is available in **environment.yml**. These dependency
files use compatible ranges rather than an archival lock. The training manifest
captures resolved versions, but a camera-ready artifact should also publish a
tested lockfile/container and CUDA/cuDNN details.

The bundled coarse defaults are:

- weights/yolov8n.pt
- weights/yolo11n.pt
- data/stop_sign.png
- data/stop_sign_uv.png
- data/pole.png
- data/backgrounds/

Binary assets are managed through Git LFS. Complete provenance, redistribution
rights, calibration, and physical metadata in
[data/DATA_CARD.md](data/DATA_CARD.md) before an artifact release.

## Training

### Experimental v2 amortized path

Copy the non-runnable template to a study-specific manifest, replace every
placeholder, provide measured calibration whose canonical SHA-256 matches each
task, and preregister all four splits. Then train one policy across the `train`
tasks:

~~~powershell
python train_amortized.py --task-manifest ./configs/paper_tasks.json --support-scenes 4 --risk-alpha 0.25 --total-steps 800000 --seed 0 --output-dir ./runs/amortized/seed_0
~~~

The output directory receives `amortized_run_manifest.json`, policy checkpoints,
and checkpoint-specific normalization state. On successful completion, the run
manifest seals offline detector-image query totals and final policy/normalizer
hashes. Run multiple independent training seeds for evidence. A non-empirical
calibration, including the template's synthetic fixture, is accepted only with
`--allow-uncalibrated-simulation`, which is a software-debug mode and cannot
support physical or measured-model claims.

Generate the finite candidate family from `development` tasks only:

~~~powershell
python tools/generate_amortized_prefixes.py --task-manifest ./configs/paper_tasks.json --run-manifest ./runs/amortized/seed_0/amortized_run_manifest.json --model ./runs/amortized/seed_0/amortized_prefix_policy_final.zip --vecnormalize ./runs/amortized/seed_0/vecnormalize_final.pkl --max-prefix-length 64 --max-development-detector-queries-per-task 10000 --out-json ./runs/amortized/seed_0/development_prefixes.json
~~~

The artifact is labeled `frozen_candidate_family_not_a_certificate`. It records
zero calibration/certification queries and does not consume those outcomes. Its
`candidate_prefixes` shape is converted by `build_risk_protocol.py`; do not edit
the converted patterns by hand. The query cap defaults to unlimited (`0`); a
paper run must choose and preregister a finite value instead of inheriting that
default.

Copy the non-runnable
[trial-inventory template](configs/risk_trial_inventory.template.json), replace
every placeholder, and build the content-bound protocol:

~~~powershell
python tools/build_risk_protocol.py --prefix-family ./runs/amortized/seed_0/development_prefixes.json --trial-inventory ./configs/paper_risk_trials.json --out ./runs/amortized/seed_0/protocol.json
~~~

Before opening outcomes, externally timestamp or register the immutable family
artifact/digest, completed inventory, and resulting protocol as one study-plan
bundle. Registering the inventory alone does not freeze the generated prefixes.

The builder rejects candidate-family self-hash mismatches, debug artifacts,
task-ID mismatches, malformed/duplicate-key JSON, and invalid protocol
structure. It does not authenticate authorship, verify files or captures behind
declared hashes, prove sample independence or prior registration, or generate
result rows.

### Legacy single-task path

The generic entry point preserves the legacy implementation module so existing
checkpoints remain loadable:

~~~powershell
python train_traffic_sign.py --data ./data --bgdir ./data/backgrounds_train --sign-profile stop --source-class "stop sign" --attack-mode disappearance --yolo-weights ./weights/yolov8n.pt --num-envs 1 --vec dummy --seed 0
~~~

The Bash launcher exposes the same objective fields:

~~~bash
bash train.sh \
  --bgdir ./data/backgrounds_train \
  --attack-mode disappearance \
  --source-class "stop sign" \
  --seed 0
~~~

The checkpoint directory receives **experiment_manifest.json**, checkpoint-
specific **vecnormalize_*_steps.pkl** files, and bounded minimal-area successful
stencils. The TensorBoard run directory receives cumulative training query
accounting. When using an in-process CUDA detector, keep
**--vec dummy --num-envs 1**; use the authenticated localhost detector server
for shared multi-environment inference.

### Targeted fine-grained speed example

This command demonstrates the required interface; the asset and weights are not
included.

~~~powershell
python train_traffic_sign.py --sign-profile custom --sign-image ./data/speed_25_day.png --sign-active-image ./data/speed_25_active.png --source-class "speed_limit_25" --attack-mode targeted_misclassification --attack-target-class "speed_limit_55" --yolo-weights ./weights/fine_grained_traffic_sign.pt --bgdir ./data/backgrounds_train --seed 0
~~~

For untargeted sign-to-sign alteration, preregister alternatives:

~~~text
--attack-mode untargeted_misclassification
--allowed-alternative-classes "speed_limit_35,speed_limit_45,speed_limit_55"
~~~

Requested classes must exist in the checkpoint label map. Numeric IDs are also
validated when a map exists.

## Evaluation

### Scene-conditioned policy evaluation

~~~powershell
python tools/eval_policy.py --model ./_runs/checkpoints/run/grid_800000_steps.zip --vecnorm ./_runs/checkpoints/run/vecnormalize_800000_steps.pkl --episodes 100 --seed 100000 --bgdir ./data/backgrounds_validation --attack-mode targeted_misclassification --source-class "speed_limit_25" --attack-target-class "speed_limit_55" --yolo-weights ./weights/fine_grained_traffic_sign.pt --out-json ./_runs/eval/summary.json --out-episodes-json ./_runs/eval/episodes.json
~~~

This evaluates an adaptive, per-scene optimizer and labels the JSON accordingly.
It does not certify a fixed physical stencil. Missing VecNormalize statistics
are a hard error.

### Frozen paint-once certification

Select a stencil using training/validation data, freeze its JSON, and evaluate
that exact cell set on a disjoint background directory and fresh RNG seeds:

~~~powershell
python tools/eval_frozen_pattern.py --pattern-json ./_runs/eval/summary.json --pattern-type selected_indices --episode-index 0 --episodes 100 --seed-base 1000000 --bgdir ./data/backgrounds_certification --allow-protocol-transfer --sign-profile custom --sign-image ./data/speed_25_day.png --sign-active-image ./data/speed_25_active.png --attack-mode targeted_misclassification --source-class "speed_limit_25" --attack-target-class "speed_limit_55" --yolo-weights ./weights/fine_grained_traffic_sign.pt --out-json ./_runs/certification/frozen_pattern.json
~~~

The output includes pattern/input hashes, every seed row, joint success, a
Wilson 95% interval, mean localized metrics, certification-only detector-image
queries, and runtime. A fresh integer seed over training backgrounds is not a
held-out dataset; use a disjoint **--bgdir**, acknowledge that distribution
change with **--allow-protocol-transfer**, and record its manifest. Geometry and
material mismatches remain hard errors even in a transfer study.

### Two-phase v2 prefix certification

The frozen evaluator produces measurements for an immutable candidate. There is
currently no automatic adapter from its summary to certification rows. Given
explicit rows in the documented schema, the separate
[risk-certification CLI](tools/certify_attack_results.py) checks a preregistered
finite family of prefixes on calibration trials, seals the smallest passing
order, and accepts only declared-disjoint final-certification trials for that
selection:

~~~powershell
python tools/certify_attack_results.py hash-plan --plan protocol.json
python tools/certify_attack_results.py calibrate --plan protocol.json --rows calibration_rows.json --out calibration_selection.json
python tools/certify_attack_results.py certify --plan protocol.json --selection calibration_selection.json --rows certification_rows.json --out certificate.json
~~~

The calibration phase allocates familywise error across prefixes, tasks, and
four outcomes; final certification allocates it across tasks and outcomes.
Those finite-sample bounds apply only to the preregistered sampled population
and sampling unit. They are not formal verification, an all-world robustness
guarantee, or evidence that any current candidate passes. Result rows must follow
the exact schema in [docs/CERTIFICATION_PROTOCOL.md](docs/CERTIFICATION_PROTOCOL.md);
the CLI rejects duplicate JSON keys and does not infer missing trials or
silently convert legacy summaries. A
selected prefix ID can contain a different pattern for each task, so it is a
sealed task-indexed family unless the plan deliberately binds one paint-once
stencil. A self-declared hash checks internal consistency. Detecting later
mutation requires comparison with a digest retained in a trusted external
record; hashes are not signatures, trusted timestamps, or proof of physical
provenance.

### Budget-matched black-box comparisons

The paper-facing native harness reconstructs a fresh, fixed-EOT oracle for each
method, rejects contract drift, and applies identical detector-image and exact
sign-alpha-pixel limits:

~~~powershell
python tools/run_budgeted_comparison.py --environment-json ./configs/paper_comparison.json --methods random_search,forward_greedy,genetic_algorithm,gaussian_es,fipatch_style_pso_proxy,cma_es --detector-query-limit 10000 --material-area-fraction 0.20 --scene-seed 1001 --optimizer-seed 7 --method-config-json ./configs/paper_budgeted_methods.json --output ./runs/comparisons/task_001_seed_7.json
~~~

Start from the deliberately non-runnable
[environment](configs/budgeted_comparison.template.json) and
[method](configs/budgeted_methods.template.json) templates. The output records
the complete candidate trace, the scalar-score winner, the best candidate that
actually has `joint_success=true`, registry fidelity labels, local detector
weight and background hashes, and the common contract fingerprint. Existing
reports are not overwritten.

This command covers one task/scene and optimizer seed; a paper needs a
preregistered task-by-seed matrix and uncertainty intervals. The binary PSO row
is only a FIPatch-family proxy. Named-paper comparisons require reviewed
external wrappers and candidate mappings; use the non-runnable
[external manifest](configs/external_baseline_manifest.template.json) and
[mapping](configs/external_candidate_mapping.template.json) templates. The
[full comparison protocol](docs/BUDGETED_COMPARISONS.md) defines query scope,
material accounting, offline-versus-online amortization cost, aggregation, and
claim boundaries.

The older **tools/run_baselines_compare.sh** remains a legacy PPO/greedy/random
convenience runner. It records queries but does not enforce the new matched
contract and must not be used for headline comparisons.

## Repository map

~~~text
train_amortized.py              experimental v2 training entry point
envs/
  attack_objective.py          pure ROI-localized objective definitions
  amortized_traffic_sign_env.py target-conditioned support-batch wrapper
  robust_objective.py          empirical tail-risk aggregation
  traffic_sign_grid_env.py     generic public environment entry point
  stop_sign_grid_env.py        implementation + compatibility import path
detectors/
  class_names.py               strict source/target label resolution
  factory.py                   YOLO, torchvision, RT-DETR, remote backends
utils/
  task_manifest.py             strict split/task contract and leakage checks
  fluorescence_transport.py    measured spectral-to-linear-RGB transport
  risk_certification.py        two-phase certification schemas and decisions
  certification_stats.py       exact binomial confidence calculations
  sign_assets.py               stop, speed-limit, and custom asset profiles
  experiment_manifest.py       hashes, versions, and resolved configuration
baselines/
  grid_utils.py                shared construction and joint evaluation
  budgeted/                    matched-budget optimizers + external contracts
tools/
  build_risk_protocol.py       prefix-family + trial-inventory binding
  certify_attack_results.py    calibration selection + final certification
  run_budgeted_comparison.py   native matched query/material comparison
  eval_policy.py               adaptive policy evaluation
  eval_frozen_pattern.py       immutable held-out stencil measurement
  generate_amortized_prefixes.py development-only v2 family generation
  replay_patterns_over_angles.py
configs/
  amortized_tasks.template.json non-runnable four-split task template
  risk_trial_inventory.template.json non-runnable risk-design template
schemas/
  fluorescence_transport_v1.schema.json
tests/
docs/
  PREFIX_VALID_AMORTIZED_METHOD.md
  fluorescence_transport.md
  CERTIFICATION_PROTOCOL.md
  NOVELTY_AND_EVALUATION.md
~~~

## Submission protocol

Before using any result in a paper:

1. Freeze the threat model, objective, label set, thresholds, material model,
   train/development/calibration/certification splits, and query budget in the
   task and certification manifests.
2. Train at least five independent policy/search seeds. Separate offline PPO
   training queries from online synthesis queries and compute amortization
   break-even points.
3. Generate and hash the entire capped candidate family using development tasks
   only; complete the trial inventory; build the risk protocol with
   `build_risk_protocol.py`; then externally register the family artifact or
   digest, inventory, and built protocol before opening outcomes.
4. Regenerate all baselines with the centralized evaluator. Treat legacy files
   and tables as invalid for the new claims.
5. Evaluate every preregistered prefix on the calibration split, seal the
   smallest passing order, and certify only that selection on disjoint trials.
6. Report simultaneous confidence bounds and raw failures, not only best runs or
   averages.
7. Validate repeated approach videos, tracking/planning consequences, clean
   utility, and adaptive defenses.
8. Release complete manifests, class maps, weight/dataset licenses, calibration
   metadata, stencils, raw detections, and table/figure scripts.

The detailed related-work boundary and experimental checklist are in
[docs/NOVELTY_AND_EVALUATION.md](docs/NOVELTY_AND_EVALUATION.md). The exact v2
method contract is in
[docs/PREFIX_VALID_AMORTIZED_METHOD.md](docs/PREFIX_VALID_AMORTIZED_METHOD.md).
