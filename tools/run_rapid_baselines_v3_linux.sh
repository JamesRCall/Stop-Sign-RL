#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Explicit revision-3 protocol; trailing arguments can select a detector,
# provide a new output folder, or resume this same frozen configuration.
exec bash tools/run_rapid_baselines_linux.sh \
  --output runs/modern_baselines_v3_rgb \
  --color-mode rgb --search-objective suppression \
  --search-areas 0.25,0.5 --circles 3 --swarm 16 --population 16 \
  --budget-policy fixed --queries 9006 --hours 13.5 "$@"
