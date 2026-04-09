#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
    echo "Expected .venv in $ROOT_DIR" >&2
    exit 1
fi

source .venv/bin/activate

mkdir -p benchmarks/results

STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_PATH="${NCU_OUTPUT_PATH:-benchmarks/results/env-profile-${STAMP}}"
LOG_FILE="$(mktemp)"

set +e
ncu \
    --target-processes all \
    --force-overwrite \
    --set none \
    --section SpeedOfLight \
    --section Occupancy \
    --section LaunchStats \
    --section SchedulerStats \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    -o "$OUTPUT_PATH" \
    python benchmarks/benchmark_env.py "$@" 2>&1 | tee "$LOG_FILE"
NCU_STATUS=${PIPESTATUS[0]}
set -e

if [[ $NCU_STATUS -ne 0 ]] && grep -q "ERR_NVGPUCTRPERM" "$LOG_FILE"; then
    echo "Nsight Compute needs access to NVIDIA GPU performance counters." >&2
    echo "See: https://developer.nvidia.com/ERR_NVGPUCTRPERM" >&2
fi

rm -f "$LOG_FILE"
exit "$NCU_STATUS"
