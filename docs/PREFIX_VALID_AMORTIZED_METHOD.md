# Prefix-valid task-amortized fluorescent stencil optimization

## Status and claim discipline

This document specifies the repository's proposed research contribution. It is
a testable method and evaluation contract, not evidence that the method works
and not a guarantee of publication. Novelty depends on the complete system and
the held-out experiments below. Individual ingredients such as reinforcement
learning, meta-learned black-box attacks, targeted patches, fluorescent traffic-
sign attacks, nested patches, and probabilistic robustness bounds all have prior
art.

The proposed contribution is the combination of three inseparable properties:

1. one optimizer is trained across source sign, target class, detector, camera,
   fluorescent material, and environment tasks;
2. it emits an irreversible sequence of ink additions for which every prefix is
   a physically fabricable stencil; and
3. a prefix is selected and evaluated with preregistered, simultaneous finite-
   sample risk bounds over a declared physical capture population.

The paper should use a descriptive name until a final title and acronym have
been checked against published and concurrent work.

## Research question

Can a single learned optimizer reuse information from previous physical patch
tasks to reduce the online query and material cost of a new targeted traffic-
sign attack, while producing an inclusion-monotone fabrication sequence whose
minimum reliable prefix is selected without test-set tuning?

This is different from training PPO separately for one sign and detector. It is
also different from calling a scene-specific action trace a universal patch.

## Task-distribution formulation

A task is

```text
task = (source sign instance, designated target, attack mode,
        detector and label map, camera/ISP calibration,
        fluorescent material batch, illumination distribution,
        background/placement distribution, constraints)
```

The policy observation must expose a versioned task descriptor, the current
stencil, and only the query feedback allowed by the threat model. Task IDs are
metadata, not a substitute for compositional conditioning. Speed values,
source/target semantics, detector family, physical calibration hashes, and
constraint values must be represented explicitly.

Training, development, calibration, and final-certification partitions are
defined in a task manifest. Sign instances, physical runs, material batches,
cameras, and background collections that are declared held out must not cross
partition boundaries. Holding out only an integer RNG seed is not task
generalization.

## Prefix-valid fabrication contract

Let `a_k` be the kth selected canonical sign-grid cell and let `P_k` be the
stencil after k actions:

```text
P_0 = empty
P_k = P_(k-1) union {a_k}
```

The following are hard invariants:

- actions use one canonical full-grid coordinate system across sign shapes;
- a cell can be added at most once;
- a later action cannot remove, move, recolor, or reduce earlier material;
- every prefix can be exported using the same physical dimensions and material;
- exact painted sign pixels, not a cell-count approximation, define area; and
- the complete ordered sequence is frozen and hashed before calibration.

Attack success is not assumed to improve monotonically with added ink. Every
candidate prefix must be evaluated. The value of monotonicity is fabrication:
an operator can stop after any action without undoing prior work.

## Support-batch optimization

At each action, the same candidate stencil is evaluated over a support batch of
matched day/active scenes for one sampled task. The action is shared across the
batch. The policy therefore optimizes one stencil over a distribution instead
of solving unrelated scene-specific episodes.

Use a lower-tail risk objective such as CVaR together with explicit constraints
for clean eligibility, inactive/day preservation, target success, and exact
material area. Log the mean, tail value, each violation rate, and the associated
dual variables separately. A scalar reward is not the reported success
predicate.

Offline meta-training queries, support/adaptation queries, generator forward
passes, calibration queries, and untouched certification queries are separate
ledger categories. Claims of query efficiency require the amortization break-
even point relative to optimizing every task independently.

## Measured fluorescence and camera model

RGB paint constants are useful for software tests but cannot support a physical
claim. Paper experiments require versioned measured inputs for:

- wavelength grid and units;
- daylight and UV illuminant spectra;
- substrate reflectance;
- material excitation/emission or measured effective response;
- material concentration/application and batch variation;
- camera RGB spectral sensitivities, exposure, and ISP/color matrix; and
- uncertainty distributions for irradiance, camera, material, and measurement.

The calibration files and their hashes are part of the task descriptor.
Synthetic fixtures must be labeled non-empirical and must never be reported as
measured fluorescence fidelity.

## Risk-limiting prefix selection

The ordered policy output defines a finite family of candidate prefixes. Freeze
that family before opening calibration results. Do not assume the candidates'
losses are monotone.

For each task and prefix, treat an independent physical run or capture cluster
as one Bernoulli unit. Video frames from the same run are not independent. Use
fixed sample sizes and one-sided exact binomial bounds for preregistered claims
such as joint success, clean eligibility, and day preservation. Allocate the
familywise error rate across every task, claim, and candidate prefix, for
example with a Bonferroni allocation.

Select the smallest-area prefix whose simultaneous bounds satisfy every task
constraint. Area is a deterministic integer-pixel/material constraint rather
than a binomial rate. Evaluate only the selected prefix on a final untouched
certification split.

The permissible statement is narrowly scoped:

> With the preregistered confidence level, the sealed task-indexed prefix
> family satisfies the declared expected-risk limits for the stated sampled
> capture population.

Use “frozen stencil” only when every task actually binds the same paint-once
physical artifact.

Do not call this an all-world guarantee, formal robustness certificate, or
guaranteed physical attack.

## Required comparisons and ablations

Use identical task splits, canonical cells, physical model, constraints, and
query/material budgets for:

- independently trained per-task PPO;
- FIPatch-style particle swarm optimization;
- PatchAttack and the closest traffic-sign RL optimizer;
- random, greedy, GA, ES, CMA-ES, and local random search;
- a task-conditioned policy without support batching;
- a support-batched policy without task conditioning;
- mean reward versus lower-tail/CVaR optimization;
- arbitrary cell sequences versus prefix-valid monotone fabrication; and
- simulated RGB constants versus measured spectral/camera uncertainty.

The executable and external-method fidelity boundaries for these rows are in
the [budgeted-comparison protocol](BUDGETED_COMPARISONS.md). A binary PSO-family
proxy is not FIPatch, and a classifier-oriented PatchAttack adaptation is not
the unmodified ECCV method.

Primary plots should show worst-task joint success against online queries and
exact material area, with offline amortization cost reported separately.

## Evidence required before a novelty claim

- one frozen policy, not a separately trained model per source-target pair;
- held-out source-target combinations and sign instances;
- at least one held-out detector family and camera/material batch;
- prefix curves and minimum selected prefixes for every task;
- complete query ledgers and amortization break-even analysis;
- independent physical capture clusters with simultaneous bounds;
- system-level tracker/planner effects and an adaptive defense evaluation; and
- a renewed related-work search immediately before submission.

Without these results, the repository contains a novel-method hypothesis and
research infrastructure, not a substantiated novelty claim.
