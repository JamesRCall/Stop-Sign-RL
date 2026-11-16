#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Same search settings/budget as v3; only IMPACT's cropped-edge rule is fixed.
exec bash tools/run_rapid_baselines_linux.sh \
  --methods IMPACT --output runs/impact_all_cells_v4 \
  --color-mode rgb --search-objective suppression \
  --search-areas 0.25,0.5 --circles 3 --swarm 16 --population 16 \
  --budget-policy fixed --queries 9006 --hours 13.5 "$@"
