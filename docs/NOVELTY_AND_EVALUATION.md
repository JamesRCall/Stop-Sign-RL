# Novelty Positioning and Evaluation Protocol

## Purpose and status

This document is a candid research-positioning guide for a security paper based
on this repository. It is not an acceptance claim. No implementation or set of
experiments can guarantee acceptance, and the current repository should not be
described as a completed speed-sign or targeted-misclassification study.

As of August 2, 2026, the USENIX Security '27 preliminary call lists Cycle 1
mandatory registration on August 18, paper submission on August 25, and
artifact submission on August 28, 2026 (all AoE). Verify the governing call
again before acting on these dates:
[USENIX Security '27 preliminary CFP](https://www.usenix.org/conference/usenixsecurity27/call-for-papers).

The repository now contains an **experimental v2 architecture** in addition to
the legacy per-scene trainer. Its software contracts make the intended study
possible; unit tests are not evidence that the method improves queries,
material, generalization, or physical robustness. No trained v2 checkpoints,
paper-ready speed-sign assets, measured calibration records, or empirical v2
tables are included. In particular:

- Current speed-sign support is **plumbing only**. The repository does not ship
  a validated fine-grained speed-sign detector, a complete speed-sign asset
  suite, or new physical speed-sign results.
- Common COCO checkpoints contain a `stop sign` category but do not distinguish
  speed limits such as 25, 35, 45, or 55. A COCO run cannot substantiate a
  fine-grained speed-sign misclassification claim.
- The legacy PPO environment clears its selected-cell mask at every reset and
  remains a per-instance optimizer. The separate v2 wrapper keeps one stencil
  across a support batch and trains one policy across declared tasks. A frozen
  v2 policy can still emit a different stencil for each held-out task; it is not
  automatically a universal paint-once patch.
- Result files, plots, checkpoints, and tables produced before the current
  localized objective, explicit class-resolution, and sign-profile changes are
  **stale for the new claims**. They must not be relabeled as targeted or
  speed-sign results. All headline experiments must be rerun from clean,
  versioned configurations.
- Supporting several detector backends is not the same as optimizing a detector
  ensemble. Use “multi-backend compatibility” unless the attack is trained
  jointly across models and evaluated on held-out models.

## Implemented v2 architecture and evidence boundary

The implemented components correspond to five auditable stages:

| Stage | Implemented software contract | What is not established |
|---|---|---|
| Preregistered tasks | The strict [task-manifest loader](../utils/task_manifest.py) hashes manifests, resolves task-relative paths, defines `train`, `development`, `calibration`, and `certification` splits, and rejects declared cross-split leakage. The [template](../configs/amortized_tasks.template.json) enumerates the required fields. | The template is non-runnable and its assets, detectors, and calibration are placeholders. A manifest does not prove the sampled tasks represent deployment. |
| Amortized target-conditioned training | [train_amortized.py](../train_amortized.py) samples only training tasks. [AmortizedTrafficSignEnv](../envs/amortized_traffic_sign_env.py) conditions observations on source/target, detector, calibration, constraints, and task features; one action is evaluated on every support scene; empirical lower-tail CVaR, constraint duals, and query counters are exposed. A completed run manifest seals offline query totals and final policy/normalizer hashes. | There is no trained-policy result, no demonstrated held-out generalization, and no demonstrated online-query or material saving. Several identities are opaque hash fingerprints; feature semantics are not validated; background/run IDs are not direct task-vector fields; and only the first support image is observed directly. Other replicas affect aggregate feedback/reward. The ledger scope must be reconciled with every external query source. |
| Prefix-valid output | Canonical cell actions only add unused cells. The environment checks strict digital inclusion. The [development-only generator](../tools/generate_amortized_prefixes.py) loads frozen artifacts, emits a capped ordered sequence with exact image-pixel area and support measurements, hashes task/sign/material/calibration descriptors and policy artifacts, and assembles a content-addressed common-order candidate family. [build_risk_protocol.py](../tools/build_risk_protocol.py) binds that family to a researcher-supplied trial inventory and validates the resulting strict plan, which must then be externally registered for a preregistration claim. | “Prefix-valid” is currently a digital invariant, not proof of physical fabricability. Generation stops at the declared prefix/query limit or environment termination, and the common family is truncated to the shortest task sequence. Per-prefix descriptors omit some physical dimensions/volume, detector/camera identity, paint, and transport-draw fields. Neither the generator nor the binding tool reads outcomes, proves prior registration, or issues a certificate. |
| Spectral/camera transport | The [v1 schema](../schemas/fluorescence_transport_v1.schema.json), [strict loader and transport](../utils/fluorescence_transport.py), and [model documentation](fluorescence_transport.md) cover wavelength grids, ambient/UV spectra, substrate, fluorescence, camera/ISP response, uncertainty draws, and input hashes. | The bundled fixture is explicitly synthetic and non-empirical. Training converts each seeded result to one opaque effective day/active 8-bit sRGB cell color per support replica; it does not render spatial spectra, geometry-dependent illumination, camera noise, or a nonlinear ISP. `measured` provenance is declared and structurally validated, not authenticated by the loader. |
| Disjoint prefix certification | Given protocol and result rows, the [two-phase CLI](../tools/certify_attack_results.py) requires complete fixed calibration rows for all declared prefixes, rejects duplicate JSON keys/non-finite constants, seals the smallest passing order, requires non-overlapping declared final sample hashes, and computes simultaneous one-sided exact binomial bounds with exact area and query checks. | No current candidate is certified. The CLI validates supplied booleans, hashes, areas, and counts; it does not run detectors, inspect content behind hashes, prove physical independence/preregistration, validate prefix-set inclusion, or automatically ingest evaluator output. Integrity hashes are not signatures, and one prefix ID may contain a different task pattern for every task. The bound is neither formal verification nor an all-world guarantee. |

These components are an integrated method hypothesis. Their combination must be
evaluated end to end, and each component must survive the ablations below,
before the paper describes the combination as a contribution.

Two researcher-supplied boundaries remain. The prefix generator consumes only
`development` tasks. The protocol builder verifies the declared canonical
family self-hash and binds that hash, task patterns, exact areas, and query
totals to a supplied trial inventory. It does not infer that inventory from
task-manifest `calibration`/`certification` entries, run detectors, or convert
captures into result rows. Those rows and their physical independence still
require an audited study pipeline; do not claim end-to-end empirical
certification merely because the schemas bind correctly. A self-declared digest
establishes internal consistency only. Tamper evidence requires comparison with
a digest retained in a trusted external record; no hash here authenticates
authorship, establishes a trusted timestamp, or verifies the physical content
named by a digest.

## Novelty assessment

The attack medium and broad premise are closely preempted. *The Fluorescent
Veil* (FIPatch), published at NeurIPS 2025 after an initial September 2024
preprint, already studies transparent fluorescent ink applied to traffic signs,
activation with ultraviolet light, digital fluorescence modeling, black-box
particle-swarm optimization, expectation over transformation (EOT), area
minimization, inactive-state stealth, hiding, creation, and misrecognition. It
evaluates ten recognition models, physical conditions, and five defenses.

Primary source:

- [The Fluorescent Veil: A Stealthy and Effective Physical Adversarial Patch Against Traffic Sign Recognition (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html)
- [FIPatch paper PDF](https://papers.nips.cc/paper_files/paper/2025/file/8e608f20d2bc14ffe312635285e0125c-Paper-Conference.pdf)
- [FIPatch preprint and submission history](https://arxiv.org/abs/2409.12394)

Consequently, fluorescent ink, UV triggering, a normally benign state,
traffic-sign hiding, misrecognition, EOT, black-box optimization, and small-area
fluorescent perturbations are not defensible standalone novelty claims.

The optimization space is also crowded:

- [PatchAttack (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5563_ECCV_2020_paper.php)
  uses reinforcement learning to optimize patch texture and position for
  targeted and untargeted black-box attacks.
- [Meta-Attack (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Meta-Attack_Class-Agnostic_and_Model-Agnostic_Physical_Adversarial_Attack_ICCV_2021_paper.html)
  formulates physical attack as few-shot learning over support/query sets and a
  target model, with generalization to novel images and DNN models.
- [Simulator Attack (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.html)
  trains a generalized simulator over attack-query tasks to reduce online
  queries against unseen target models. It is not a fluorescent physical-patch
  method, but it preempts a generic “learn offline to save black-box queries”
  claim.
- [Simultaneously Optimizing Perturbations and Positions for Black-box Adversarial Patch Attacks](https://arxiv.org/abs/2212.12995)
  ([IEEE TPAMI DOI](https://doi.org/10.1109/TPAMI.2022.3231886))
  applies an RL framework to targeted and untargeted traffic-sign attacks on
  TT100K using YOLO surrogates and a NanoDet target.
- [RPAttack (ICME 2021)](https://arxiv.org/abs/2103.12469) selects and refines
  sparse high-impact regions and jointly attacks YOLO and Faster R-CNN.
- [IMPACT (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html)
  jointly optimizes patch mask, content, shape, location, and number using
  gradient-free search and produces physically applicable irregular patches.
- [Budget-Aware Adaptive Adversarial Patches](https://arxiv.org/abs/2606.18318)
  (the arXiv record reports acceptance to ICIP 2026) couples contextual Thompson
  sampling and NES-style updates and grows a patch when progress stalls under
  query and visual-footprint budgets.
- [Targeted Physical Evasion Attacks in the Near-Infrared Domain (NDSS 2026)](https://www.ndss-symposium.org/ndss-paper/targeted-physical-evasion-attacks-in-the-near-infrared-domain/)
  ([paper PDF](https://www.ndss-symposium.org/wp-content/uploads/2026-s1568-paper.pdf))
  studies square “manypixel” perturbations, targeted stop/speed scenarios,
  fine-grained sign datasets, EOT, moving-vehicle trials, defenses, and direct
  comparisons of local random search, genetic algorithms, evolutionary
  strategies, PSO, and random search.

The following table is a claim boundary, not an exhaustive patent or literature
search. “Preempted” means the repository must not present that element alone as
its novelty.

| Candidate claim | Primary prior work that preempts or narrows it | Permissible positioning |
|---|---|---|
| First fluorescent/UV, normally benign traffic-sign patch | [FIPatch](https://papers.nips.cc/paper_files/paper/2025/hash/8e608f20d2bc14ffe312635285e0125c-Abstract-Conference.html) | Preempted. Treat fluorescence and UV activation as the medium, not the contribution. |
| First hiding, creation, misrecognition, EOT, or reduced-area fluorescent patch | [FIPatch paper](https://papers.nips.cc/paper_files/paper/2025/file/8e608f20d2bc14ffe312635285e0125c-Paper-Conference.pdf) | Preempted. Use the repository's localized predicates only as clearer evaluation definitions. |
| First triggered or transient physical patch | [TPatch (USENIX Security 2023)](https://www.usenix.org/conference/usenixsecurity23/presentation/zhu) and [SLAP (USENIX Security 2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/lovisotto) | Preempted. Dual-state triggering is not standalone novelty. |
| First RL or black-box targeted/untargeted patch | [PatchAttack (ECCV 2020)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5563_ECCV_2020_paper.php) | Preempted. PPO is an implementation choice unless cross-task amortization is demonstrated. |
| First RL traffic-sign patch or targeted traffic-sign RL attack | [Wei et al., TPAMI](https://doi.org/10.1109/TPAMI.2022.3231886) | Preempted. Fine-grained task conditioning and held-out task performance require a direct comparison. |
| First meta-learned, amortized, class-agnostic, or model-agnostic physical attack | [Meta-Attack (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_Meta-Attack_Class-Agnostic_and_Model-Agnostic_Physical_Adversarial_Attack_ICCV_2021_paper.html) | Preempted. Support/query task construction and novel-image/model adaptation are prior art; task amortization alone is not the contribution. |
| First offline learned model that reduces queries to unseen black-box targets | [Simulator Attack (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Ma_Simulating_Unknown_Target_Models_for_Query-Efficient_Black-Box_Attacks_CVPR_2021_paper.html) | Preempted at the generic query-amortization level. Demonstrate measured savings against this broader design pattern and count all offline queries. |
| First sparse, irregular, minimum-area, or minimum-material patch | [RPAttack](https://arxiv.org/abs/2103.12469), [IMPACT](https://papers.nips.cc/paper_files/paper/2025/hash/8172ca14a9ec80a3409113d1d1f8bc42-Abstract-Conference.html), and FIPatch | Preempted or heavily narrowed. The testable distinction is an irreversible ordered prefix family plus risk-limited selection, not sparsity alone. |
| First budget-aware, footprint-aware, or adaptively growing black-box patch | [Budget-Aware Adaptive Adversarial Patches (arXiv/ICIP 2026)](https://arxiv.org/abs/2606.18318) | Preempted. A growing patch under query/footprint budgets is not enough; compare directly at matched budgets and discuss concurrent overlap. |
| First targeted stop/speed or fine-grained speed-sign physical scenario | [NDSS 2026 near-IR study](https://www.ndss-symposium.org/ndss-paper/targeted-physical-evasion-attacks-in-the-near-infrared-domain/) | Preempted. Speed-sign compatibility is an evaluation domain, not novelty. |
| First fluorescence or camera simulation | FIPatch already includes digital fluorescence modeling | Not supportable. The v2 distinction is a strict measured-input and uncertainty contract whose fidelity must be quantified. |
| New confidence-bound method | Exact binomial intervals and Bonferroni control are standard statistical tools | Do not claim statistical novelty. Their role is to prevent prefix-selection leakage and scope the empirical statement. |
| First combination of amortization, prefix-valid fabrication, measured transport, and disjoint risk certification | Not established by the cited set | This remains a hypothesis, not a “first” claim. Any defensible distinction must hinge on the combined ordered irreversible fluorescent prefix family, conditioning across held-out source-target-detector-camera/material tasks, and disjoint simultaneous finite-sample selection—not meta-learning or patch growth alone. Re-run a systematic search immediately before submission. |

Adding PPO or speed-sign assets to FIPatch is therefore unlikely to be viewed as
a sufficient contribution. The paper needs a research question that makes the
learned policy necessary and tests a capability absent from per-instance search.

## Defensible research framing

A concise method label is **target-conditioned, task-amortized, prefix-valid
fluorescent stencil optimization**. The exact proposed claim sentence is:

> We study whether a target-conditioned optimizer trained across a
> preregistered task distribution can reduce online query and material costs by
> emitting inclusion-monotone fluorescent-stencil prefixes, with the smallest
> qualifying prefix selected on calibration trials and evaluated on disjoint
> certification trials under measured spectral/camera inputs and simultaneous
> finite-sample risk bounds.

This wording does not assert that a reduction or a novelty result has already
been observed. It makes each part falsifiable and keeps the statistical scope
at sampled-population risk rather than universal robustness.

This framing is credible only if the policy is trained across a distribution of
tasks and then evaluated on held-out tasks. A PPO agent trained separately for
one stop-sign image and one detector is an expensive per-instance search method;
it is not evidence of amortization or generalization.

The v2 code implements the method-side contracts, but the following evidence
and ablations are required before the corresponding paper clauses are asserted:

| Claim clause | Required evidence | Decisive comparison or ablation |
|---|---|---|
| One optimizer amortizes across tasks | Train one frozen policy over multiple source-target, sign, detector, camera, material, and scene tasks; evaluate without retraining on held-out sign instances and source-target combinations, including at least one held-out detector family and camera/material batch. Report offline and online queries separately and the break-even deployment count. | Per-task PPO and per-task PSO/ES/CMA-ES at equal online query, material, and wall-clock budgets; v2 without task conditioning; v2 with shuffled or ID-only conditioning; zero-, few-, and full-online-query operating points. |
| Support batching learns distributional rather than scene-specific stencils | Show that one action sequence is shared across each support batch and improves held-out task distributions, not only its training scenes. Report mean and lower-tail performance and every constraint violation. | Support size 1 versus larger fixed sizes; task-conditioned policy without support batching; mean reward versus lower-tail CVaR; fixed versus learned constraint duals. |
| Prefix-valid output improves the query/material operating frontier | Freeze and hash full ordered sequences, evaluate every prefix because success may be nonmonotone, and report prefix curves and exact painted-pixel area for every task. | Unrestricted add/remove or final-set optimization versus irreversible additions at matched queries and final area; fixed prefix lengths versus risk-limited selection. |
| Measured spectral/camera transport improves physical fidelity | Collect versioned empirical spectra for illuminants, substrate, material batches, camera sensitivities, exposure, and ISP; report held-out digital-to-physical RGB and detector-prediction error with uncertainty coverage across batches and cameras. | Legacy constant RGB; measured point estimate without uncertainty; full bounded uncertainty; leave-one-camera and leave-one-material-batch-out tests. The bundled synthetic fixture is excluded from this evidence. |
| Disjoint selection supports a scoped risk statement | Preregister tasks, independent capture-cluster sampling units, prefix family, sample sizes, thresholds, area cap, query totals, and familywise alpha. Release calibration selection and untouched certification rows with disjoint hashes and simultaneous bounds. | Validation-tuned or uncorrected prefix selection versus corrected calibration selection followed by final holdout; coverage simulation for the stated sampling design; sensitivity to task and claim multiplicity. Do not present this as a new statistical estimator. |
| The integrated combination is more than its components | Run an end-to-end factorial study using identical task splits, constraints, calibration, and budgets, with enough independent training and physical-replicate seeds for uncertainty estimates. | Remove each of task conditioning, support batching, prefix validity, measured uncertainty, and disjoint selection in turn; report interactions and failure cases, not only the full model. |
| The work is systems-security relevant | Demonstrate consequences for tracking/planning in the critical distance range and evaluate adaptive defenses with clean-utility, latency, and hardware costs. | Per-frame detector ASR versus tracker/planner outcomes; undefended versus adaptive defense; component-only versus system-level evaluation. |

An alternative, and potentially stronger security-paper direction, is a
measurement and defense study: reproduce FIPatch, quantify its
simulation-to-physical and component-to-system gaps, identify when fluorescent
attacks fail across materials and cameras, and develop an adaptive defense. That
direction must still compare with the system-level findings in
[SysAdv (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Does_Physical_Adversarial_Example_Really_Matter_to_Autonomous_Driving_Towards_ICCV_2023_paper.html).

## Claims that must not be made

Do not claim any of the following:

- first UV or fluorescent adversarial traffic-sign patch;
- first invisible, triggerable, or normally benign physical patch;
- first physical stop-sign disappearance attack;
- first physical traffic-sign misclassification or designated-target attack;
- first stop-to-speed, speed-to-stop, or speed-limit-sign attack;
- first RL adversarial patch or first RL traffic-sign attack;
- first grid, sparse, minimum-area, or minimum-material patch merely because the
  implementation selects cells;
- first black-box, EOT-robust, transferable, or multi-detector physical patch;
- universal patch, if a new stencil is generated for each evaluation scene;
- real-world speed-sign support, until it is demonstrated using an appropriate
  model and physical evaluation;
- multi-detector optimization, if experiments only select one backend at a
  time; or
- guaranteed USENIX acceptance.

If a scoped “first” claim is ultimately used, it must name every essential
qualifier and be rechecked immediately before submission. Prefer the exact
question-form claim above; do not turn it into a result statement until the
required evidence exists.

## Threat models: do not mix adaptive and paint-once results

The paper must choose and clearly separate two valid but different threat
models.

### Per-instance optimizer

The attacker may optimize a different stencil for each sign instance, source
and target class, or deployment setting. Every target-model interaction used to
produce that stencil is charged as an online query. Evaluation may generate a
new stencil per test task, but must report the distribution of queries, runtime,
and material for those tasks. This setting evaluates an optimizer, not a
universal patch.

### Task-amortized optimizer

One frozen policy is trained across the task-manifest training distribution and
then receives an unseen task descriptor. It may emit a task-specific ordered
stencil and may use a preregistered number of support/adaptation queries. Charge
all such target-model queries online, report zero-query and adapted modes
separately, and include the offline-training cost in the amortization break-even
analysis. This evaluates cross-task optimizer reuse; it still does not establish
one universal stencil.

### Frozen paint-once stencil

The attacker selects and physically applies one stencil before deployment. The
same exported stencil must then be evaluated across all held-out digital scenes,
models, distances, angles, cameras, and physical trials. No cell selection,
threshold tuning, color tuning, or early stopping may use certification or
physical-test results. This setting evaluates a universal or distributionally
robust patch.

Results from these settings must be shown in separate tables. Per-scene adaptive
success must never be reported as fixed-stencil physical robustness.

## Precise localized objectives

Let `B` be the known or annotated sign bounding box, `s` the source class, `t`
the designated target class, `eta` the localization IoU threshold, `tau_s` the
source-suppression threshold, and `tau_t` the alternative/target-confidence
threshold. Only detections whose boxes overlap `B` by at least `eta` are eligible
to support a sign attack claim. A top detection elsewhere in the scene is not
evidence of sign misclassification.

Evaluate matched inactive/day and active/UV views with the same background,
placement, and geometric transform. Count an attack opportunity only when the
inactive view is correctly localized and classified as `s` at a preregistered
confidence threshold. Report this clean-eligibility rate rather than silently
discarding failed baselines.

Use the following distinct predicates:

### Source-class evasion

No localized detection of class `s` exceeds `tau_s` in the active view. This is
what source-confidence suppression establishes. It must not be called complete
object disappearance when another sign class remains localized at `B`.

### Object hiding or disappearance

No localized traffic-sign detection, or no localized detector objectness score,
exceeds the preregistered threshold. This requires an explicit traffic-sign
class set or access to detector objectness. Report it separately from
source-class evasion.

### Untargeted misclassification

A localized class in a preregistered traffic-sign label set, different from
`s`, is the winning localized prediction with confidence at least `tau_t`, and
the source is suppressed below `tau_s`. On a general COCO detector, an unrelated
localized COCO label should not automatically count as a semantically valid
traffic-sign alteration.

### Targeted misclassification

The designated class `t` is the winning localized prediction with confidence at
least `tau_t`, and the source is suppressed below `tau_s`. Merely decreasing
`p(s)` is not a targeted attack.

### Joint paired-state success

The inactive view remains correctly localized as `s`, while the matched active
view satisfies the selected predicate. Across `K` EOT samples, success requires
a preregistered fraction `rho` of joint successes. Report inactive preservation,
active objective success, and joint success separately.

For video, additionally report consecutive successful frames, successful time
within the system-critical distance range, target-label stability, and tracker
state. Frame-average ASR alone can hide failures that allow a tracker or planner
to recover.

## Fine-grained speed-sign requirement

Speed-limit compatibility requires a label space with distinct speed semantics.
Use at least one full-scene fine-grained detector and one two-stage recognition
pipeline:

- [TT100K: Traffic-Sign Detection and Classification in the Wild (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.html)
  or Mapillary traffic-sign data for detection and class labels;
- GTSRB for European classifier experiments; and
- LISA or another licensed North American dataset for U.S.-style speed limits.

The detector checkpoint, complete `id -> class name` map, dataset split, clean
accuracy/mAP, and license must be versioned with the artifact. Runs must fail
closed when a requested source or target label is absent; substituting a COCO id
is invalid.

Evaluate safety-relevant source/target pairs rather than pooling all labels:

- lower speed -> higher speed (unsafe acceleration);
- higher speed -> lower speed (unexpected braking);
- STOP -> speed limit (ignored stop); and
- speed limit -> STOP (unexpected stop).

Use sign masks derived from alpha or segmentation so circular, rectangular,
triangular, and octagonal signs share the same algorithm without hand-coded
geometry. Physical trials must use lawfully obtained replicas in a controlled
area, not modified public road signs.

## Four-way separation and frozen-prefix risk certification

Use four disjoint data roles, matching the task-manifest vocabulary:

1. **Policy-training split:** tasks used to update policy parameters and
   constraint duals.
2. **Development split:** tasks used for architecture choices, hyperparameters,
   reward/constraint definitions, checkpoint choice, and early stopping.
3. **Calibration split:** fixed trials used only to evaluate the already frozen
   candidate-prefix family and select the smallest qualifying order.
4. **Certification split:** untouched trials used once to evaluate only the
   sealed selection under the final simultaneous confidence allocation.

The builder automates only the development-family-to-inventory binding; the
four-role split remains a study-design responsibility. It requires the trial
inventory's task IDs to equal the development candidate-family task IDs and
takes calibration/certification trial hashes from that inventory. It does not
import task-manifest rows labeled `calibration` or `certification`. Report the
actual binding used by the experiment rather than claiming those manifest rows
enforced the risk phases.

The [task-manifest loader](../utils/task_manifest.py) rejects overlap in every
declared leakage key. The [certification protocol](CERTIFICATION_PROTOCOL.md)
also rejects duplicate or overlapping sample hashes across phases and against
declared training/development exclusions. Neither mechanism can detect an
undeclared duplicate or semantically equivalent capture, so dataset provenance
and a manual leakage audit remain required.

For a paint-once claim, export a canonical stencil containing the sign-relative
cell coordinates, colors/alphas, grid definition, sign mask hash, and checkpoint
hash before certification. Evaluate exactly that artifact on the certification
split and then fabricate the same artifact for physical testing.

For an amortized-optimizer claim, freeze the policy and allow it to generate a
stencil for each certification task, but charge every task-specific detector
query. Report zero-query, few-query, and full-budget operating points.

Before opening calibration outcomes, freeze and hash the complete ordered prefix
family, task definitions, exact-area cap, outcome thresholds, sampling unit,
fixed sample sizes, query totals, and familywise alpha. Build the combined
protocol and externally register it with the family artifact/digest and trial
inventory; the self-hashes alone do not establish preregistration. Evaluate
every prefix on the same preregistered calibration trials; attack success is not
assumed monotone in prefix length. The implemented CLI allocates calibration
error over

```text
prefixes x tasks x {clean eligibility, attack success, day preservation, joint success}
```

and chooses the smallest preregistered order whose one-sided exact
Clopper--Pearson lower bounds and deterministic area checks pass for every task.
It seals that selection by hash. Final certification accepts only the sealed
prefix ID/order, requires a disjoint complete trial list, reconciles
detector-image query counts, and allocates final error over
`tasks x 4 outcomes`. A protocol prefix can contain a different pattern for each
task, so describe the result as a sealed task-indexed prefix family unless the
experiment truly uses one paint-once stencil.

Use [tools/certify_attack_results.py](../tools/certify_attack_results.py) for
this statistical layer. It consumes schema-constrained protocol/result rows; it
does not create physical captures or silently translate legacy evaluator
summaries. Its JSON reader rejects duplicate object keys and non-finite
constants, but the declared hashes and Boolean outcomes remain researcher-
supplied facts rather than authenticated measurements.

Additional certification requirements:

- no train/development/calibration/certification leakage in signs, backgrounds,
  cameras, material batches, physical-run groups, or excluded sample hashes;
- fixed confidence, IoU, EOT success-rate, and area thresholds;
- at least five independent training/search seeds, with 95% confidence
  intervals;
- per-class and per-condition results in addition to macro averages;
- failures and clean-ineligible cases retained in released records; and
- a final untouched physical or captured-video holdout.

The resulting statement is limited to the declared sampled population and
independent sampling unit. It is not formal verification, an all-world physical
guarantee, or a guarantee that a current repository candidate passes.

## Equal-query and equal-material baselines

The main comparison must include strong derivative-free methods, not only
random and greedy search:

- FIPatch PSO with its paired-state and area loss;
- Meta-Attack or the closest faithful few-shot physical-attack adaptation;
- a learned simulator/prior baseline for the query-amortization claim where the
  detector-feedback interface permits a faithful implementation;
- the 2026 budget-aware adaptive-growth method at equal query and footprint
  budgets;
- local random search, GA, ES, PSO, and random search following the NDSS 2026
  comparison;
- the repository's random and greedy cell-selection baselines;
- CMA-ES or a similarly strong continuous/discrete black-box optimizer;
- PatchAttack or a faithful RL patch baseline;
- the Wei et al. simultaneous RL optimizer;
- an IMPACT-style gradient-free sparse-mask optimizer where applicable;
- RPAttack only as a separately labeled white-box sparse-mask upper bound; and
- a white-box EOT optimizer as an upper bound, clearly separated from black-box
  results.

The implemented [matched-budget harness](BUDGETED_COMPARISONS.md) currently
runs native random, forward greedy, binary GA, Gaussian ES, a plainly labeled
binary PSO-family proxy, and the installed reference `cma` package against one
fixed task/scene at a time. It hashes the common objective, detector weights,
background collection, selected scene, sign assets, fixed EOT seeds, and exact
material costs; rejects budget or contract drift; and records the full trace.
The proxy must not be reported as FIPatch. FIPatch, PatchAttack, the 2026
budget-adaptive method, per-task PPO, Wei et al., Meta-Attack, Simulator Attack,
and IMPACT have fail-closed external slots, not bundled results. Each requires a
pinned wrapper, a declared/preregistered mapping into the binary-grid candidate
space, separate offline-query disclosure where relevant, and independent
semantic review before any named-method equivalence claim.

Use identical sign masks, paint parameterization, transformations, thresholds,
area caps, and source/target pairs. For every method, plot:

- joint ASR versus target-model queries;
- joint ASR versus painted sign area and measured material volume;
- ASR versus wall-clock time;
- inactive preservation versus active ASR; and
- the Pareto frontier over queries, material, and success.

Query accounting must be explicit. One detector evaluation of one image counts
as one query even when images are batched. Day and UV evaluations both count.
Queries across `K` transforms and across ensemble members are summed. Report:

- offline surrogate/policy-training queries;
- online target-model queries per generated stencil;
- total queries for the reported experiment;
- inference hardware and wall-clock time; and
- amortized cost after 1, 10, 100, and 1,000 deployments.

Without this accounting, PPO can appear query-efficient by hiding its cost in
offline training.

## Physical evaluation

The physical protocol should match or exceed the exact FIPatch and NDSS 2026
baselines. At minimum vary:

- ambient illumination measured in lux, including day, twilight, and night;
- UV wavelength, electrical power, irradiance at the sign, beam width, lamp-to-
  sign distance, and trigger duration;
- camera-to-sign distance and horizontal/vertical viewing angle;
- vehicle speed and motion blur using repeated approach videos;
- phone, dashcam, and automotive-grade cameras with different exposure/HDR and
  optical-filter behavior;
- multiple physical signs or replicas, ink brands, colors, batches, application
  methods, and operators;
- headlights, weather, surface contamination, and partial occlusion; and
- aging, UV exposure, rain/cleaning, and material durability.

Measure the ink's excitation and emission spectra or a per-wavelength effective
response, ambient and UV irradiance, substrate reflectance, camera RGB
sensitivity, exposure, and ISP matrix. Record batch/camera/irradiance
uncertainty. The implemented [spectral transport contract](fluorescence_transport.md)
validates and hashes these inputs and deterministically produces day and
triggered linear RGB for a recorded uncertainty seed. Report held-out
digital-to-physical prediction error and uncertainty coverage, not only attack
success. Use independent fabrication replicates and confidence intervals.

The repository's
[synthetic fixture](../data/synthetic/fluorescence_transport_v1.synthetic.json)
is hand-constructed for tests and API examples. It must not appear in a physical
result, calibration-fidelity table, or paper claim. Paper-facing training rejects
it unless an explicit debug override is supplied.

Inactive-state stealth requires evidence. Suitable evidence includes calibrated
color difference under normal illumination, surface reflectance measurements,
and, with appropriate ethics review, a preregistered human-subject study. Images
selected by the authors are not sufficient evidence of invisibility.

[GhostStripe (MobiSys 2024)](https://arxiv.org/abs/2407.07510) is a useful
standard for reporting stable behavior across a moving vehicle's frame
sequence. [Invisible Reflections (NDSS 2024)](https://www.ndss-symposium.org/ndss-paper/invisible-reflections-leveraging-infrared-laser-reflections-to-target-traffic-sign-perception/)
is a useful standard for multi-camera physical evaluation and an attack-specific
defense.

## System-level evaluation

Integrate the perception output with detection tracking and planning in a
reproducible stack such as CARLA plus an open driving stack. The system study
must define the distance range in which a sign prediction affects braking or
speed control and measure:

- track creation, label stability, deletion, and recovery;
- time-to-first and duration of successful attack within the critical range;
- STOP-rule violations;
- incorrect commanded speed or acceleration;
- unexpected or emergency braking; and
- benign driving utility and false interventions.

This is essential because [SysAdv](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Does_Physical_Adversarial_Example_Really_Matter_to_Autonomous_Driving_Towards_ICCV_2023_paper.html)
found that representative component-level STOP-sign attacks did not
automatically cause system-level violations. Do not infer a crash or traffic
violation solely from a low detector confidence.

## Defense evaluation

A security paper should evaluate adaptive defenses, not only generic image
preprocessing. Include:

- the five defense families evaluated by FIPatch;
- localized patch defenses such as SentiNet/PatchCleanser where their
  assumptions apply;
- the segmentation-based defense proposed in the NDSS 2026 near-IR work;
- temporal consistency and map/route consistency checks;
- multi-camera or multi-sensor consistency;
- optical or spectral filtering, with clean-utility and night-performance cost;
  and
- an adaptive attacker that knows the defense.

Report clean accuracy/mAP, false-positive rate, latency, hardware cost, and
attack ASR after defense. A defense that destroys fine-grained speed numerals or
substantially harms nighttime perception is not operationally successful.

## Reporting and reproducibility checklist

Before submission, release or provide to reviewers:

- exact commit identifier and environment lockfile;
- completed amortized run manifests with offline-query totals and final
  policy/normalizer hashes;
- detector weights, label maps, dataset manifests, and licenses;
- the canonical and source hashes of the four-way task manifest;
- train/development/calibration/certification split membership and leakage audit;
- measured spectral/camera calibration files, provenance, canonical/source
  hashes, and uncertainty seeds/draws;
- all thresholds and success predicates in one machine-readable config;
- seeds, query counters, wall-clock measurements, raw per-frame detections, and
  failure cases;
- exported frozen stencils and fabrication dimensions;
- the development-generated candidate-family artifact, its canonical hash, the
  trial inventory, and the builder-produced protocol that embeds the family
  hash, with the complete bundle's externally registered digest or timestamp;
- raw physical videos with calibration metadata;
- the preregistered certification plan, calibration rows, sealed selection,
  final rows, certificate, and reconciled query ledger;
- scripts that regenerate every table and figure; and
- an ethics, safety, and responsible-disclosure statement.

The current [USENIX Security '27 preliminary CFP](https://www.usenix.org/conference/usenixsecurity27/call-for-papers)
allows 13 pages of body text, requires an Open Science Appendix, strongly
encourages an Ethics Appendix, and provides a three-day artifact-submission
grace period. It also warns that ML work must establish systems-security
relevance. Recheck the final governing CFP before submission; attacker
definition, threat surface, generality, practicality, ethics, and artifact
readiness should be treated as core study requirements rather than appendix
cleanup.

## Additional primary prior art

- [TPatch: A Triggered Physical Adversarial Patch (USENIX Security 2023)](https://www.usenix.org/conference/usenixsecurity23/presentation/zhu)
- [SLAP: Improving Physical Adversarial Examples with Short-Lived Adversarial Perturbations (USENIX Security 2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/lovisotto)
- [Physical Adversarial Examples for Object Detectors (WOOT 2018)](https://www.usenix.org/conference/woot18/presentation/eykholt)
- [Robust Physical-World Attacks on Deep Learning Visual Classification (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Eykholt_Robust_Physical-World_Attacks_CVPR_2018_paper.html)
- [ShapeShifter: Robust Physical Adversarial Attack on Faster R-CNN](https://arxiv.org/abs/1804.05810)
- [Fooling the Eyes of Autonomous Vehicles (NDSS 2022)](https://www.ndss-symposium.org/wp-content/uploads/2022-130-paper.pdf)
- [Natural Light Can Also Be Dangerous (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/Hsiao_Natural_Light_Can_Also_Be_Dangerous_Traffic_Sign_Misinterpretation_Under_WACV_2024_paper.html)
- [SwitchPatch preprint](https://arxiv.org/abs/2506.08482), which studies a
  static patch with trigger-selectable attack objectives
- [Adversarial Retroreflective Patch preprint](https://arxiv.org/abs/2511.10050),
  which includes stop and speed-limit signs, dynamic physical trials, a user
  study, and an attack-specific defense
- [AdvAD preprint](https://arxiv.org/abs/2604.23105), which jointly optimizes
  transferable physical patches across object detectors

Preprints should be labeled as such in the related-work section, but they remain
relevant to novelty and reviewer expectations.
