#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  echo "Usage: bash scripts/run_experiment_pipeline.sh <step1|step2|all_steps> <dataset> [limit]"
}

read_pipeline_config() {
  python -m src.pipeline.read_pipeline_config --config "${CONFIG_PATH}" "$@"
}

run_step1() {
  echo "=== START step1 fa_exp | dataset=${DATASET} limit=${LIMIT:-all} ==="

  for SPLIT in "${SPLITS[@]}"; do
    echo "=== Running step1 fa_exp for dataset=${DATASET} split=${SPLIT} limit=${LIMIT:-all} ==="

    python scripts/build_manifest.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --dataset "${DATASET}" \
      --splits "${SPLIT}" \
      "${LIMIT_ARGS[@]}"

    python scripts/generate_oracle_variants.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_asr_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --manifest "artifacts/manifests/${SPLIT}.jsonl" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_asr_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --manifest "artifacts/manifests/${SPLIT}_oracle_variants.jsonl"

    python scripts/summarize_results.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step1 \
      --inputs \
      "exp/step1/decodes/${SPLIT}__${ASR_TAG}.jsonl" \
      "exp/step1/decodes/${SPLIT}_oracle_variants__${ASR_TAG}.jsonl" \
      --output-prefix "${DATASET}_${SPLIT}"
  done

  python -m src.analysis.render_step1_views \
    --config "${CONFIG_PATH}" \
    --logging-config "${LOGGING_CONFIG_PATH}" \
    --exp-subdir step1 \
    --dataset "${DATASET}"

  echo "[done] step1 fa_exp completed successfully for dataset=${DATASET} limit=${LIMIT:-all}"
}

run_step2() {
  local policy_tag="silero_vad_${POLICY_NAME}"
  local step1_decode_dir="${PROJECT_ROOT}/exp/step1/decodes"
  local step2_decode_dir="${PROJECT_ROOT}/exp/step2/decodes"
  local manifest_dir="${PROJECT_ROOT}/artifacts/manifests"

  echo "=== START step2 apply_vad | dataset=${DATASET} limit=${LIMIT:-all} ==="

  for SPLIT in "${SPLITS[@]}"; do
    echo "=== Running step2 apply_vad for dataset=${DATASET} split=${SPLIT} limit=${LIMIT:-all} ==="

    local manifest_path="${manifest_dir}/${SPLIT}.jsonl"
    local baseline_decode_path="${step1_decode_dir}/${SPLIT}__${ASR_TAG}.jsonl"

    if [[ ! -f "${manifest_path}" ]]; then
      echo "Missing manifest for step2: ${manifest_path}"
      echo "Run step1 first."
      exit 1
    fi

    if [[ ! -f "${baseline_decode_path}" ]]; then
      echo "Missing baseline decode for step2: ${baseline_decode_path}"
      echo "Run step1 first."
      exit 1
    fi

    local step2_baseline_input="${baseline_decode_path}"
    if [[ -n "${LIMIT}" ]]; then
      mkdir -p "${step2_decode_dir}"
      step2_baseline_input="${step2_decode_dir}/${SPLIT}__${ASR_TAG}__step2_limit${LIMIT}.jsonl"
      python -m src.eval.filter_decode_by_manifest \
        --manifest "${manifest_path}" \
        --input "${baseline_decode_path}" \
        --output "${step2_baseline_input}" \
        --limit "${LIMIT}"
    fi

    python scripts/run_silero_vad_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --policy default \
      --exp-subdir step2 \
      --manifest "${manifest_path}" \
      "${LIMIT_ARGS[@]}"

    python scripts/run_silero_vad_decode.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --policy fa_informed \
      --exp-subdir step2 \
      --manifest "${manifest_path}" \
      "${LIMIT_ARGS[@]}"

    python scripts/summarize_results.py \
      --config "${CONFIG_PATH}" \
      --logging-config "${LOGGING_CONFIG_PATH}" \
      --exp-subdir step2 \
      --inputs \
      "${step2_baseline_input}" \
      "exp/step2/decodes/${SPLIT}__silero_vad_default__${ASR_TAG}.jsonl" \
      "exp/step2/decodes/${SPLIT}__${policy_tag}__${ASR_TAG}.jsonl" \
      --output-prefix "${DATASET}_${SPLIT}"
  done

  python -m src.analysis.render_step2_views \
    --config "${CONFIG_PATH}" \
    --logging-config "${LOGGING_CONFIG_PATH}" \
    --exp-subdir step2 \
    --dataset "${DATASET}"

  echo "[done] step2 apply_vad completed successfully for dataset=${DATASET} limit=${LIMIT:-all}"
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage
  exit 1
fi

STAGE="${1}"
DATASET="${2}"
LIMIT="${3:-}"

CONFIG_PATH="${PROJECT_ROOT}/conf/base.yaml"
LOGGING_CONFIG_PATH="${PROJECT_ROOT}/conf/logging.yaml"

if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
else
  LIMIT_ARGS=()
fi

readarray -t SPLITS < <(read_pipeline_config --dataset "${DATASET}" --field splits)
ASR_TAG="$(read_pipeline_config --field asr_tag)"
POLICY_NAME="$(read_pipeline_config --field vad_policy_name)"

case "${STAGE}" in
  step1)
    run_step1
    ;;
  step2)
    run_step2
    ;;
  all_steps)
    run_step1
    run_step2
    ;;
  *)
    echo "Invalid stage: ${STAGE}"
    usage
    exit 1
    ;;
esac
