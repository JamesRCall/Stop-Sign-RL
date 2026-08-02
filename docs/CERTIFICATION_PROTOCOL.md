# Frozen-prefix risk certification

`tools/certify_attack_results.py` implements a two-phase, fixed-sample
evaluation protocol. It produces finite-sample sampled-population risk bounds;
it does not provide formal verification or an all-world guarantee.

`tools/build_risk_protocol.py` provides the preceding structural binding step. It
combines a frozen candidate-family artifact with a researcher-supplied trial
inventory, requires exact task-ID agreement, converts the generated pattern
areas into the risk schema, imports query totals, and records the family SHA-256
in the protocol. The
[inventory template](../configs/risk_trial_inventory.template.json) contains
placeholder task, detector, sample, exclusion, and query values. It is neither a
runnable study design nor evidence of preregistration.

## Builder and integrity boundary

For a paper-facing build, the prefix family must have an internally consistent
declared canonical self-hash, the expected method and schema, `development`
generation status, and no non-empirical or incomplete-ledger debug flag. The
builder then:

- requires the candidate family and trial inventory to contain exactly the same
  task-ID set;
- copies every task-specific pattern hash and exact integer-pixel area from the
  candidate family;
- imports offline-training and prefix-generation query totals from the family;
- imports development-selection and expected calibration/certification query
  totals from the inventory;
- embeds `prefix_family_sha256`; and
- runs the same strict protocol parser used by calibration and certification.

This is a content-consistency binding, not authenticated provenance. A
self-declared SHA-256 can be recomputed after content changes; detecting such a
change requires comparison with a digest retained in a trusted external record.
A digest is not a signature or trusted timestamp. The builder does not establish
that the family, inventory, or combined protocol existed before outcomes,
authenticate an author, inspect a detector or sample behind a declared digest,
prove physical independence, or run an evaluation. Externally register or
timestamp the built protocol together with the family artifact/digest and
inventory before opening calibration outcomes.

The candidate generator uses task-manifest `development` tasks. The inventory
defines calibration and certification trial hashes for those same task IDs; the
builder does not import task-manifest rows labeled `calibration` or
`certification`. Do not cite those unused manifest rows as enforcement of the
statistical split.

## Statistical claim

For every preregistered task and Boolean outcome, the tool computes a one-sided
exact Clopper--Pearson lower confidence bound. Equal Bonferroni allocation
controls the simultaneous familywise error rate. Calibration allocates over

```
number of prefixes x number of tasks x 4 outcomes
```

and final certification allocates over

```
number of tasks x 4 outcomes.
```

The four required outcomes are `clean_eligibility`, `attack_success`,
`day_preservation`, and `joint_success`. They are fail-closed:

- `attack_success` and `day_preservation` cannot be true when
  `clean_eligibility` is false.
- `joint_success` must equal clean eligibility AND attack success AND day
  preservation AND the exact material-area constraint.

Material area is compared as integer pixel counts by cross multiplication. It
does not consume statistical alpha.

The binomial interpretation requires independent Bernoulli sampling within
each task. A scene cluster can be the sampling unit, but then the reported
probability is the probability that a fresh cluster passes the complete
cluster predicate. Correlated transforms must not be flattened and described
as independent samples.

## Two-phase selection

The protocol requires a result row for every candidate on exactly the same
declared calibration trials. The CLI checks the complete supplied row set; it
does not run the detector. Selection never stops after the first passing
candidate and never assumes attack success is monotone as material is added. It
seals the candidate with the smallest declared `order` that passes all task,
outcome, and area constraints.

Final certification accepts only that sealed prefix ID/order and a separate
list of certification samples whose hashes are disjoint from training,
development, and calibration samples. The fixed list must be complete. Missing,
additional, duplicate, nonfinite, or malformed outcomes are rejected rather
than omitted.

A prefix can contain a different pattern hash for every task. Unless every task
deliberately references one physical pattern, call the output a sealed
task-indexed prefix family rather than one frozen stencil. Protocol validation
checks nondecreasing exact area by order, not inclusion of the underlying cell
sets; the generated family supplies the stronger digital-inclusion invariant.

Hash disjointness detects only identical declared digests. It cannot detect
semantically duplicated captures stored under different bytes or establish that
the named sampling units are independent.

## Protocol schema

Protocol and result objects reject unknown fields. CLI JSON readers also reject
duplicate object keys and non-finite constants. Fractions use reduced integer
numerator and denominator pairs to make thresholds and comparisons unambiguous.

```json
{
  "schema_version": 1,
  "protocol_id": "paper_protocol_v1",
  "prefix_family_sha256": "<SHA-256 of the frozen candidate-family artifact>",
  "estimand": {
    "population_description": "held-out deployment scenes from the stated sampling frame",
    "sampling_unit": "independent_scene_transform_pair",
    "task_aggregation": "all_tasks",
    "fixed_sample_size": true
  },
  "familywise_alpha": {
    "calibration": {"numerator": 1, "denominator": 20},
    "certification": {"numerator": 1, "denominator": 20}
  },
  "claims": {
    "joint_success": {"numerator": 4, "denominator": 5},
    "attack_success": {"numerator": 4, "denominator": 5},
    "day_preservation": {"numerator": 19, "denominator": 20},
    "clean_eligibility": {"numerator": 9, "denominator": 10}
  },
  "area_cap": {"numerator": 1, "denominator": 5},
  "tasks": [
    {
      "task_id": "stop_to_speed25_yolo",
      "detector_sha256": "<64 lowercase hex characters>",
      "calibration_trials": [
        {"trial_id": "cal_0001", "sample_sha256": "<64 lowercase hex characters>"}
      ],
      "certification_trials": [
        {"trial_id": "cert_0001", "sample_sha256": "<64 lowercase hex characters>"}
      ]
    }
  ],
  "prefixes": [
    {
      "prefix_id": "cells_08",
      "order": 8,
      "task_patterns": [
        {
          "task_id": "stop_to_speed25_yolo",
          "pattern_sha256": "<64 lowercase hex characters>",
          "area": {"selected_pixels": 1200, "sign_pixels": 10000}
        }
      ]
    }
  ],
  "excluded_sample_sha256": {
    "training": [],
    "development": []
  },
  "query_accounting": {
    "offline_training_detector_image_queries": 0,
    "development_selection_detector_image_queries": 0,
    "prefix_generation_detector_image_queries": 0,
    "expected_calibration_detector_image_queries": 4,
    "expected_certification_detector_image_queries": 4
  }
}
```

Each result row has the following exact shape:

```json
{
  "task_id": "stop_to_speed25_yolo",
  "prefix_id": "cells_08",
  "trial_id": "cal_0001",
  "sample_sha256": "<hash copied from the protocol>",
  "pattern_sha256": "<hash copied from the protocol>",
  "detector_sha256": "<hash copied from the protocol>",
  "outcomes": {
    "joint_success": true,
    "attack_success": true,
    "day_preservation": true,
    "clean_eligibility": true
  },
  "area": {"selected_pixels": 1200, "sign_pixels": 10000},
  "detector_image_queries": 4
}
```

Calibration result roots contain `schema_version`, `protocol_sha256`,
`phase: "calibration"`, and `rows`. Certification result roots additionally
contain the `selection_sha256` emitted by calibration and use
`phase: "certification"`.

## Commands

```powershell
python tools/build_risk_protocol.py `
  --prefix-family frozen_prefixes.json `
  --trial-inventory trial_inventory.json `
  --out protocol.json

python tools/certify_attack_results.py hash-plan --plan protocol.json

python tools/certify_attack_results.py calibrate `
  --plan protocol.json `
  --rows calibration_rows.json `
  --out calibration_selection.json

python tools/certify_attack_results.py certify `
  --plan protocol.json `
  --selection calibration_selection.json `
  --rows certification_rows.json `
  --out certificate.json
```

Build the protocol from the completed inventory, then externally register the
protocol together with the family artifact/digest and inventory before
collecting or opening calibration outcomes. Registering only the inventory does
not freeze the generated prefix family. `hash-plan` validates the protocol and
prints its canonical semantic hash; that command is not itself a trusted
timestamp.

The CLIs refuse to overwrite existing outputs. Offline-training and
prefix-generation counts are imported from the content-bound candidate family.
Development-selection and expected calibration/certification counts are
researcher declarations in the inventory. Observed calibration and
certification counts are summed from the supplied result-row ledger and must
match the expected phase totals exactly. Calibration/certification reports also
record raw input-file hashes and an outer transport-envelope hash, but none of
these hashes authenticates the underlying measurements.
