#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for crater_profile in small medium large; do
  echo "Running short MGM example for crater profile: ${crater_profile}"
  "${SCRIPT_DIR}/run_short_example.sh" "${crater_profile}"
done
