#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRATER_PROFILE="${1:-medium}"

python "${PROJECT_ROOT}/main_simulation_harmonized.py" \
  --crater-profile "${CRATER_PROFILE}" \
  --output-root "${PROJECT_ROOT}/outputs/example_short_${CRATER_PROFILE}" \
  --total-time 10100 \
  --dt 100 \
  --dt-flex 1000 \
  --flexure-outside-fill-mode zero \
  --drainage-only \
  --no-save-figures \
  --no-progress
