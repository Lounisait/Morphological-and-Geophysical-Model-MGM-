#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PROJECT_ROOT}/main_simulation_harmonized.py" \
  --surface-profile "${PROJECT_ROOT}/data/surf_profile_2000.csv" \
  --output-root "${PROJECT_ROOT}/outputs/example_short" \
  --total-time 10100 \
  --dt 100 \
  --dt-flex 1000 \
  --drainage-only \
  --no-save-figures \
  --no-progress
