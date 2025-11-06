#!/usr/bin/env bash
set -euo pipefail
# Re-run command captured at 2025-11-07T01:53:23.780794
python sEMG_SVM.py --data ./dataset --test-size 0.2 --kernel rbf --grid
