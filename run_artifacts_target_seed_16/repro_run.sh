#!/usr/bin/env bash
set -euo pipefail
# Re-run command captured at 2025-11-07T01:59:15.223776
python sEMG_SVM.py --data ./dataset --test-size 0.2 --kernel rbf --grid
