#!/usr/bin/env bash
set -euo pipefail

OPENSLR_BASE_URL="https://www.openslr.org/resources/12"
DATA_ROOT="data"
SPLITS=(dev-clean dev-other test-clean test-other)
EXTRACT=1

usage() {
  cat <<'EOF'
Usage: bash scripts/data/download_librispeech.sh [options]

Download LibriSpeech subsets from OpenSLR SLR12.

Options:
  --data-root PATH        Directory that will contain LibriSpeech/ (default: data)
  --splits LIST          Space-separated split list, e.g. "dev-clean test-clean"
  --no-extract           Download archives only
  -h, --help             Show this help message

Default splits match the KCC 2026 experiments:
  dev-clean dev-other test-clean test-other
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --splits)
      read -r -a SPLITS <<< "$2"
      shift 2
      ;;
    --no-extract)
      EXTRACT=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "${DATA_ROOT}"

download_file() {
  local url="$1"
  local output_path="$2"

  if [[ -f "${output_path}" ]]; then
    echo "[skip] archive exists: ${output_path}"
    return
  fi

  echo "[download] ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - --output "${output_path}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${output_path}" "${url}"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi
}

for split in "${SPLITS[@]}"; do
  archive_name="${split}.tar.gz"
  archive_path="${DATA_ROOT}/${archive_name}"
  download_file "${OPENSLR_BASE_URL}/${archive_name}" "${archive_path}"

  if [[ "${EXTRACT}" -eq 1 ]]; then
    if [[ -d "${DATA_ROOT}/LibriSpeech/${split}" ]]; then
      echo "[skip] extracted split exists: ${DATA_ROOT}/LibriSpeech/${split}"
    else
      echo "[extract] ${archive_path}"
      tar -xzf "${archive_path}" -C "${DATA_ROOT}"
    fi
  fi
done

echo "[done] LibriSpeech download step complete."
