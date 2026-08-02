#!/usr/bin/env bash
# Run the implemented paper-facing experiment pipeline without streaming large
# training logs into an interactive terminal.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: bash tools/run_paper_experiments.sh [--init-configs]"
  echo
  echo "Primary environment variables:"
  echo "  RUN_ID=usenix27_linux_01       unique result bundle name"
  echo "  RESULTS=/absolute/path         default: <repo>/runs/<RUN_ID>"
  echo "  RESUME=1                       skip already validated stages"
  echo "  RESTART_PARTIAL=1              preserve/rename and restart partial seed runs"
  echo "  PYTHON_BIN=/path/to/python     default: <repo>/.venv/bin/python"
  echo "  TRAIN_SEEDS='0 1 2 3 4'"
  echo "  QUERY_BUDGETS='2000 5000 10000'"
  echo "  SCENE_SEEDS='1001 1002 1003 1004 1005'"
  echo "  OPTIMIZER_SEEDS='7 17 29 43 71'"
  echo "  RUN_TESTS=0|1, RUN_TRAINING=0|1, RUN_PREFIXES=0|1"
  echo "  RUN_RISK=0|1, RUN_COMPARISONS=0|1, ARCHIVE_RESULTS=0|1"
  echo
  echo "Certification is run only when REGISTERED_PLAN=1 and both"
  echo "CALIBRATION_ROWS and CERTIFICATION_ROWS name existing result files."
}

copy_if_missing() {
  local source="$1"
  local destination="$2"
  if [[ ! -e "$destination" ]]; then
    mkdir -p "$(dirname "$destination")"
    cp -- "$source" "$destination"
    echo "Created $destination"
  else
    echo "Kept existing $destination"
  fi
}

init_configs() {
  copy_if_missing configs/amortized_tasks.template.json configs/paper_tasks.json
  copy_if_missing configs/risk_trial_inventory.template.json configs/paper_risk_trials.json
  copy_if_missing configs/budgeted_methods.template.json configs/paper_budgeted_methods.json
  local name
  for name in \
    stop_disappearance_a10 \
    stop_disappearance_a20 \
    speed25_untargeted_a10 \
    speed25_untargeted_a20 \
    speed25_to55_a10 \
    speed25_to55_a20
  do
    copy_if_missing \
      configs/budgeted_comparison.template.json \
      "configs/paper_comparisons/${name}.json"
  done
  echo
  echo "Configuration templates are ready. Replace every placeholder with"
  echo "licensed assets, fine-grained weights, empirical calibration, disjoint"
  echo "splits, real hashes, fixed trials, and exact query totals before running."
}

case "${1:-}" in
  --init-configs)
    init_configs
    exit 0
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  "") ;;
  *)
    usage >&2
    exit 2
    ;;
esac

RUN_ID="${RUN_ID:-usenix27_linux_01}"
RESULTS="${RESULTS:-$ROOT/runs/$RUN_ID}"
RESUME="${RESUME:-0}"
RESTART_PARTIAL="${RESTART_PARTIAL:-0}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
TASK_MANIFEST="${TASK_MANIFEST:-$ROOT/configs/paper_tasks.json}"
RISK_INVENTORY="${RISK_INVENTORY:-$ROOT/configs/paper_risk_trials.json}"
METHOD_CONFIG="${METHOD_CONFIG:-$ROOT/configs/paper_budgeted_methods.json}"
DEV_SEED="${DEV_SEED:-424242}"
PRIMARY_SEED="${PRIMARY_SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
TRAIN_STEPS="${TRAIN_STEPS:-800000}"
MAX_PREFIX_LENGTH="${MAX_PREFIX_LENGTH:-64}"
MAX_DEVELOPMENT_QUERIES="${MAX_DEVELOPMENT_QUERIES:-10000}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_PREFIXES="${RUN_PREFIXES:-1}"
RUN_RISK="${RUN_RISK:-1}"
RUN_COMPARISONS="${RUN_COMPARISONS:-1}"
ARCHIVE_RESULTS="${ARCHIVE_RESULTS:-1}"
REGISTERED_PLAN="${REGISTERED_PLAN:-0}"
CALIBRATION_ROWS="${CALIBRATION_ROWS:-}"
CERTIFICATION_ROWS="${CERTIFICATION_ROWS:-}"
REGISTRATION_REFERENCE="${REGISTRATION_REFERENCE:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

IFS=' ' read -r -a TRAIN_SEED_LIST <<< "${TRAIN_SEEDS:-0 1 2 3 4}"
IFS=' ' read -r -a QUERY_BUDGET_LIST <<< "${QUERY_BUDGETS:-2000 5000 10000}"
IFS=' ' read -r -a SCENE_SEED_LIST <<< "${SCENE_SEEDS:-1001 1002 1003 1004 1005}"
IFS=' ' read -r -a OPTIMIZER_SEED_LIST <<< "${OPTIMIZER_SEEDS:-7 17 29 43 71}"

METHODS="random_search,forward_greedy,genetic_algorithm,gaussian_es,fipatch_style_pso_proxy,cma_es"
COMPARISON_ROWS=(
  "stop_disappearance|$ROOT/configs/paper_comparisons/stop_disappearance_a10.json|0.10"
  "stop_disappearance|$ROOT/configs/paper_comparisons/stop_disappearance_a20.json|0.20"
  "speed25_untargeted|$ROOT/configs/paper_comparisons/speed25_untargeted_a10.json|0.10"
  "speed25_untargeted|$ROOT/configs/paper_comparisons/speed25_untargeted_a20.json|0.20"
  "speed25_to55|$ROOT/configs/paper_comparisons/speed25_to55_a10.json|0.10"
  "speed25_to55|$ROOT/configs/paper_comparisons/speed25_to55_a20.json|0.20"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

progress() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_file() {
  [[ -f "$1" ]] || die "required file does not exist: $1"
}

require_no_placeholders() {
  local file="$1"
  local pattern='REPLACE|replace-|NON-RUNNABLE|data/synthetic|0000000000000000|1111111111111111|2222222222222222|3333333333333333|4444444444444444'
  local matches
  matches="$(grep -Ein "$pattern" "$file" || true)"
  if [[ -n "$matches" ]]; then
    echo "$matches" >&2
    die "paper-invalid placeholders remain in $file"
  fi
}

record_timing() {
  local label="$1"
  local started="$2"
  local ended="$3"
  local exit_code="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$started" "$ended" "$((ended - started))" "$exit_code" \
    >> "$RESULTS/provenance/timings.tsv"
}

run_logged() {
  local label="$1"
  local logfile="$2"
  shift 2
  local donefile="${logfile}.done"
  if [[ "$RESUME" == "1" && -f "$donefile" ]]; then
    progress "SKIP $label (completion marker exists)"
    return 0
  fi
  mkdir -p "$(dirname "$logfile")"
  progress "START $label; log=$logfile"
  local started ended exit_code
  started="$(date +%s)"
  set +e
  "$@" > "$logfile" 2>&1
  exit_code=$?
  set -e
  ended="$(date +%s)"
  record_timing "$label" "$started" "$ended" "$exit_code"
  if (( exit_code != 0 )); then
    echo "--- last 100 log lines: $logfile ---" >&2
    tail -n 100 "$logfile" >&2 || true
    die "$label failed with exit code $exit_code"
  fi
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$donefile"
  progress "DONE $label in $((ended - started)) seconds"
}

json_field_equals() {
  local path="$1"
  local dotted="$2"
  local expected="$3"
  "$PYTHON_BIN" - "$path" "$dotted" "$expected" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for component in sys.argv[2].split("."):
    value = value[component]
raise SystemExit(0 if str(value) == sys.argv[3] else 1)
PY
}

require_command git
require_command grep
require_command sha256sum
require_command tar
[[ -x "$PYTHON_BIN" ]] || die "Python environment is unavailable: $PYTHON_BIN"

require_file "$TASK_MANIFEST"
require_file "$RISK_INVENTORY"
require_file "$METHOD_CONFIG"
require_no_placeholders "$TASK_MANIFEST"

for row in "${COMPARISON_ROWS[@]}"; do
  IFS='|' read -r _task config _area <<< "$row"
  require_file "$config"
  require_no_placeholders "$config"
done

if [[ -e "$RESULTS" && "$RESUME" != "1" ]]; then
  die "result directory already exists; choose a new RUN_ID or set RESUME=1: $RESULTS"
fi
mkdir -p \
  "$RESULTS/provenance" \
  "$RESULTS/inputs/comparisons" \
  "$RESULTS/tests" \
  "$RESULTS/amortized" \
  "$RESULTS/risk" \
  "$RESULTS/comparisons" \
  "$RESULTS/tables"

if [[ ! -f "$RESULTS/provenance/timings.tsv" ]]; then
  printf 'stage\tstarted_epoch\tended_epoch\telapsed_seconds\texit_code\n' \
    > "$RESULTS/provenance/timings.tsv"
fi

progress "Recording provenance"
git rev-parse HEAD > "$RESULTS/provenance/git-commit.txt"
git status --porcelain=v1 > "$RESULTS/provenance/git-status.txt"
git lfs ls-files --long > "$RESULTS/provenance/git-lfs.txt" 2>&1 || true
"$PYTHON_BIN" -m pip freeze --all > "$RESULTS/provenance/pip-freeze.txt"
uname -a > "$RESULTS/provenance/uname.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$RESULTS/provenance/nvidia-smi.txt" 2>&1 || true
fi
{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'results=%s\n' "$RESULTS"
  printf 'train_seeds=%s\n' "${TRAIN_SEED_LIST[*]}"
  printf 'development_seed=%s\n' "$DEV_SEED"
  printf 'primary_seed=%s\n' "$PRIMARY_SEED"
  printf 'query_budgets=%s\n' "${QUERY_BUDGET_LIST[*]}"
  printf 'scene_seeds=%s\n' "${SCENE_SEED_LIST[*]}"
  printf 'optimizer_seeds=%s\n' "${OPTIMIZER_SEED_LIST[*]}"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'device=%s\n' "$DEVICE"
} > "$RESULTS/provenance/run-config.txt"

cp -- "$TASK_MANIFEST" "$RESULTS/inputs/paper_tasks.json"
cp -- "$RISK_INVENTORY" "$RESULTS/inputs/paper_risk_trials.json"
cp -- "$METHOD_CONFIG" "$RESULTS/inputs/paper_budgeted_methods.json"
for row in "${COMPARISON_ROWS[@]}"; do
  IFS='|' read -r _task config _area <<< "$row"
  cp -- "$config" "$RESULTS/inputs/comparisons/$(basename "$config")"
done

run_logged "dependency check" "$RESULTS/tests/pip-check.log" \
  "$PYTHON_BIN" -m pip check
run_logged "runtime dependency inventory" "$RESULTS/tests/runtime-dependencies.log" \
  "$PYTHON_BIN" -c \
  'import importlib.metadata as m, torch; [print(x, m.version(x)) for x in ("torch", "torchvision", "stable-baselines3", "sb3-contrib", "ultralytics", "cma")]; print("cuda", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'

run_logged "task manifest validation" "$RESULTS/tests/task-manifest.log" \
  "$PYTHON_BIN" -c \
  'import sys; from utils.task_manifest import TASK_SPLITS, load_task_manifest; m=load_task_manifest(sys.argv[1]); print(m.source_sha256); print(m.canonical_sha256); [print(s, [t.task_id for t in m.tasks_for_split(s)]) for s in TASK_SPLITS]' \
  "$TASK_MANIFEST"

if [[ "$RUN_TESTS" == "1" ]]; then
  run_logged "compileall" "$RESULTS/tests/compileall.log" \
    "$PYTHON_BIN" -m compileall -q \
    train_amortized.py train_traffic_sign.py envs baselines detectors utils tools tests
  run_logged "ruff critical checks" "$RESULTS/tests/ruff.log" \
    "$PYTHON_BIN" -m ruff check . --select E9,F63,F7,F82
  run_logged "pytest" "$RESULTS/tests/pytest.log" \
    "$PYTHON_BIN" -m pytest -q --basetemp "$RESULTS/tests/pytest-tmp"
  run_logged "training environment check" "$RESULTS/tests/environment-check.log" \
    "$PYTHON_BIN" -u train_amortized.py \
    --task-manifest "$TASK_MANIFEST" \
    --support-scenes 4 \
    --risk-alpha 0.25 \
    --seed "${TRAIN_SEED_LIST[0]}" \
    --output-dir "$RESULTS/tests/environment-check" \
    --check-env
fi

COMMON_TRAIN_ARGS=(
  --task-manifest "$TASK_MANIFEST"
  --support-scenes 4
  --risk-alpha 0.25
  --required-support-success-rate 0.80
  --required-clean-eligible-rate 0.80
  --required-day-preservation-rate 0.80
  --dual-learning-rate 0.05
  --total-steps "$TRAIN_STEPS"
  --n-steps 1024
  --batch-size 256
  --learning-rate 0.0002
  --ent-coef 0.001
  --save-freq 100000
)

if [[ "$RUN_TRAINING" == "1" ]]; then
  for seed in "${TRAIN_SEED_LIST[@]}"; do
    run="$RESULTS/amortized/seed_${seed}"
    manifest="$run/amortized_run_manifest.json"
    if [[ -f "$manifest" ]] && json_field_equals "$manifest" run_status completed; then
      progress "SKIP training seed $seed (completed manifest exists)"
      continue
    fi
    if [[ -e "$run" ]]; then
      if [[ "$RESTART_PARTIAL" == "1" ]]; then
        partial="${run}.partial_$(date -u +%Y%m%dT%H%M%SZ)"
        mv -- "$run" "$partial"
        progress "Preserved partial training run at $partial"
      else
        die "partial training directory exists; set RESTART_PARTIAL=1 to preserve and restart it: $run"
      fi
    fi
    mkdir -p "$run"
    run_logged "training seed $seed" "$run/training.log" \
      "$PYTHON_BIN" -u train_amortized.py \
      "${COMMON_TRAIN_ARGS[@]}" \
      --seed "$seed" \
      --output-dir "$run"
    require_file "$manifest"
    json_field_equals "$manifest" run_status completed || \
      die "training seed $seed did not seal a completed manifest"
  done
fi

if [[ "$RUN_PREFIXES" == "1" ]]; then
  for seed in "${TRAIN_SEED_LIST[@]}"; do
    run="$RESULTS/amortized/seed_${seed}"
    family="$run/development_prefixes.json"
    require_file "$run/amortized_run_manifest.json"
    require_file "$run/amortized_prefix_policy_final.zip"
    require_file "$run/vecnormalize_final.pkl"
    if [[ -f "$family" ]] && \
      json_field_equals "$family" status frozen_candidate_family_not_a_certificate; then
      progress "SKIP prefix generation seed $seed (validated family exists)"
      continue
    fi
    run_logged "prefix generation seed $seed" "$run/prefix-generation.log" \
      "$PYTHON_BIN" -u tools/generate_amortized_prefixes.py \
      --task-manifest "$TASK_MANIFEST" \
      --run-manifest "$run/amortized_run_manifest.json" \
      --model "$run/amortized_prefix_policy_final.zip" \
      --vecnormalize "$run/vecnormalize_final.pkl" \
      --max-prefix-length "$MAX_PREFIX_LENGTH" \
      --max-development-detector-queries-per-task "$MAX_DEVELOPMENT_QUERIES" \
      --seed "$DEV_SEED" \
      --device "$DEVICE" \
      --out-json "$family"
    json_field_equals "$family" status frozen_candidate_family_not_a_certificate || \
      die "prefix family for seed $seed has an unexpected status"
  done
fi

PLAN="$RESULTS/risk/protocol.json"
PRIMARY_RUN="$RESULTS/amortized/seed_${PRIMARY_SEED}"
FAMILY="$PRIMARY_RUN/development_prefixes.json"
if [[ "$RUN_RISK" == "1" ]]; then
  require_file "$FAMILY"
  risk_placeholders="$(grep -Ein 'REPLACE|replace-|NON-RUNNABLE|0000000000000000|1111111111111111|2222222222222222|3333333333333333|4444444444444444' "$RISK_INVENTORY" || true)"
  if [[ -n "$risk_placeholders" ]]; then
    {
      echo "The risk inventory still contains placeholders."
      echo "Complete it from the frozen primary family, then rerun this driver with RESUME=1."
      echo
      echo "$risk_placeholders"
    } > "$RESULTS/risk/RISK_INVENTORY_REQUIRED.txt"
    progress "SKIP risk protocol (complete $RISK_INVENTORY, then use RESUME=1)"
  else
    rm -f "$RESULTS/risk/RISK_INVENTORY_REQUIRED.txt"
      if [[ ! -f "$PLAN" ]]; then
        run_logged "build risk protocol" "$RESULTS/risk/build-protocol.log" \
          "$PYTHON_BIN" -u tools/build_risk_protocol.py \
          --prefix-family "$FAMILY" \
          --trial-inventory "$RISK_INVENTORY" \
          --out "$PLAN"
      else
        progress "SKIP risk protocol build (protocol exists)"
      fi
      run_logged "hash risk protocol" "$RESULTS/risk/protocol-canonical-sha256.txt" \
        "$PYTHON_BIN" tools/certify_attack_results.py hash-plan --plan "$PLAN"
      sha256sum "$FAMILY" "$RISK_INVENTORY" "$PLAN" \
        > "$RESULTS/risk/protocol-file-hashes.txt"
      {
        echo "Externally register the prefix family, inventory, protocol, and hashes."
        echo "Do not open calibration outcomes before registration."
      } > "$RESULTS/risk/REGISTRATION_REQUIRED.txt"

      if [[ -n "$CALIBRATION_ROWS" || -n "$CERTIFICATION_ROWS" ]]; then
        [[ "$REGISTERED_PLAN" == "1" ]] || \
          die "result rows were supplied but REGISTERED_PLAN is not 1"
        [[ -n "$REGISTRATION_REFERENCE" ]] || \
          die "REGISTRATION_REFERENCE is required when running certification"
        require_file "$CALIBRATION_ROWS"
        require_file "$CERTIFICATION_ROWS"
        rm -f "$RESULTS/risk/CERTIFICATION_ROWS_REQUIRED.txt"
        printf '%s\n' "$REGISTRATION_REFERENCE" \
          > "$RESULTS/risk/registration-reference.txt"
        selection="$RESULTS/risk/calibration-selection.json"
        certificate="$RESULTS/risk/certificate.json"
        if [[ ! -f "$selection" ]]; then
          run_logged "calibration selection" "$RESULTS/risk/calibration.log" \
            "$PYTHON_BIN" tools/certify_attack_results.py calibrate \
            --plan "$PLAN" \
            --rows "$CALIBRATION_ROWS" \
            --out "$selection"
        fi
        if json_field_equals "$selection" selection.status selected; then
          if [[ ! -f "$certificate" ]]; then
            progress "START final certification"
            started="$(date +%s)"
            set +e
            "$PYTHON_BIN" tools/certify_attack_results.py certify \
              --plan "$PLAN" \
              --selection "$selection" \
              --rows "$CERTIFICATION_ROWS" \
              --out "$certificate" \
              > "$RESULTS/risk/certification.log" 2>&1
            certification_exit=$?
            set -e
            ended="$(date +%s)"
            record_timing "final certification" "$started" "$ended" "$certification_exit"
            if (( certification_exit != 0 && certification_exit != 2 )); then
              tail -n 100 "$RESULTS/risk/certification.log" >&2 || true
              die "certification crashed with exit code $certification_exit"
            fi
            printf '%s\n' "$certification_exit" \
              > "$RESULTS/risk/certification-exit-code.txt"
            progress "DONE final certification with exit code $certification_exit"
          fi
        else
          echo "No prefix passed calibration; certification was not run." \
            > "$RESULTS/risk/NO_PREFIX_PASSED.txt"
        fi
      else
        {
          echo "No certification rows were supplied."
          echo "The repository validates rows but does not generate detector/capture rows."
          echo "Set CALIBRATION_ROWS, CERTIFICATION_ROWS, REGISTERED_PLAN=1, and"
          echo "REGISTRATION_REFERENCE only after the immutable plan is registered."
        } > "$RESULTS/risk/CERTIFICATION_ROWS_REQUIRED.txt"
      fi
  fi
fi

if [[ "$RUN_COMPARISONS" == "1" ]]; then
  for row in "${COMPARISON_ROWS[@]}"; do
    IFS='|' read -r task config area <<< "$row"
    area_tag="a${area/./p}"
    for query_budget in "${QUERY_BUDGET_LIST[@]}"; do
      for scene_seed in "${SCENE_SEED_LIST[@]}"; do
        for optimizer_seed in "${OPTIMIZER_SEED_LIST[@]}"; do
          output="$RESULTS/comparisons/$task/$area_tag/q$query_budget/scene${scene_seed}_opt${optimizer_seed}.json"
          log="${output%.json}.log"
          if [[ -f "$output" ]] && \
            json_field_equals "$output" protocol_version budgeted-black-box-v2; then
            progress "SKIP comparison $task/$area_tag/q$query_budget/$scene_seed/$optimizer_seed"
            continue
          fi
          if [[ -e "$output" ]]; then
            die "invalid partial comparison output exists: $output"
          fi
          run_logged \
            "comparison $task $area_tag q$query_budget scene$scene_seed opt$optimizer_seed" \
            "$log" \
            "$PYTHON_BIN" -u tools/run_budgeted_comparison.py \
            --environment-json "$config" \
            --methods "$METHODS" \
            --detector-query-limit "$query_budget" \
            --material-area-fraction "$area" \
            --scene-seed "$scene_seed" \
            --optimizer-seed "$optimizer_seed" \
            --method-config-json "$METHOD_CONFIG" \
            --output "$output"
        done
      done
    done
  done
fi

progress "Generating CSV summaries"
"$PYTHON_BIN" - "$RESULTS" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
tables = root / "tables"
tables.mkdir(parents=True, exist_ok=True)

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

training_rows = []
prefix_rows = []
for run_manifest in sorted((root / "amortized").glob("seed_*/amortized_run_manifest.json")):
    record = json.loads(run_manifest.read_text(encoding="utf-8"))
    seed = record.get("config", {}).get("seed")
    training_rows.append({
        "seed": seed,
        "run_status": record.get("run_status"),
        "policy_steps": record.get("query_accounting", {}).get("policy_environment_steps"),
        "offline_detector_image_queries": record.get("query_accounting", {}).get("offline_training_detector_image_queries"),
        "policy_sha256": record.get("final_artifacts", {}).get("policy_sha256"),
        "vecnormalize_sha256": record.get("final_artifacts", {}).get("vecnormalize_sha256"),
    })
    family_path = run_manifest.parent / "development_prefixes.json"
    if family_path.is_file():
        family = json.loads(family_path.read_text(encoding="utf-8"))
        for task in family.get("tasks", []):
            prefix_rows.append({
                "seed": seed,
                "task_id": task.get("task_id"),
                "sequence_length": task.get("sequence_length"),
                "detector_image_queries": task.get("detector_image_queries"),
                "query_budget_exhausted": task.get("query_budget_exhausted"),
                "artifact_sha256": family.get("artifact_sha256"),
            })

comparison_rows = []
for path in sorted((root / "comparisons").rglob("*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("protocol_version") != "budgeted-black-box-v2":
        continue
    invocation = record["invocation"]
    budget = invocation["budget"]
    for result in record["results"]:
        successful = result.get("best_successful")
        best = result.get("best")
        comparison_rows.append({
            "report": str(path.relative_to(root / "comparisons")),
            "method": result["method"],
            "status": result["status"],
            "scene_seed": invocation["scene_seed"],
            "optimizer_seed": invocation["optimizer_seed"],
            "query_limit": budget["detector_query_limit"],
            "material_pixel_limit": budget["material_pixel_limit"],
            "requested_area_fraction": invocation.get("material_area_fraction_requested"),
            "queries_used": result["detector_queries_used"],
            "evaluated_candidates": result["evaluated_candidates"],
            "any_joint_success": int(successful is not None),
            "successful_score": successful["score"] if successful else "",
            "successful_material_pixels": successful["selected_material_pixels"] if successful else "",
            "successful_material_fraction": successful["material_fraction"] if successful else "",
            "best_score": best["score"] if best else "",
            "best_joint_success": int(best["joint_success"]) if best else "",
            "implementation": result["implementation"],
            "fidelity": result["fidelity"],
        })

write_csv(tables / "training_queries.csv", training_rows)
write_csv(tables / "prefix_families.csv", prefix_rows)
write_csv(tables / "budgeted_results.csv", comparison_rows)

summary = {
    "completed_training_runs": sum(row["run_status"] == "completed" for row in training_rows),
    "prefix_task_rows": len(prefix_rows),
    "comparison_method_rows": len(comparison_rows),
    "limitations": [
        "Native comparison rows do not include exact FIPatch or PatchAttack wrappers.",
        "The proposed amortized policy is not executed through the native matched-budget CLI.",
        "Certification rows must be produced by an external audited detector/capture runner.",
    ],
}
(tables / "run_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

progress "Hashing result bundle"
manifest_tmp="${RESULTS}.sha256.tmp"
find "$RESULTS" -type f ! -name results-manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$manifest_tmp"
mv -- "$manifest_tmp" "$RESULTS/results-manifest.sha256"

if [[ "$ARCHIVE_RESULTS" == "1" ]]; then
  progress "Archiving result bundle"
  archive="${RESULTS}.tar.gz"
  if [[ -e "$archive" ]]; then
    if [[ "$RESUME" == "1" ]]; then
      previous="${RESULTS}.previous_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
      mv -- "$archive" "$previous"
      if [[ -e "${archive}.sha256" ]]; then
        mv -- "${archive}.sha256" "${previous}.sha256"
      fi
      progress "Preserved previous archive: $previous"
    else
      die "archive already exists: $archive"
    fi
  fi
  tar -czf "$archive" -C "$(dirname "$RESULTS")" "$(basename "$RESULTS")"
  sha256sum "$archive" > "${archive}.sha256"
  progress "Archive written: $archive"
fi

progress "Paper experiment driver completed"
echo "Results: $RESULTS"
echo "Summary: $RESULTS/tables/run_summary.json"
echo "Timings: $RESULTS/provenance/timings.tsv"
