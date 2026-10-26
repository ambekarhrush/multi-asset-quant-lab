#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 scripts/01_signal_research.py
python3 scripts/02_risk_premia.py
python3 scripts/03_options_overlays.py
python3 -m pytest -q tests
python3 scripts/build_report.py
python3 scripts/build_readme_figures.py
