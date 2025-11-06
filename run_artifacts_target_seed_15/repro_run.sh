#!/usr/bin/env bash
set -euo pipefail
# Re-run command captured at 2025-11-07T01:57:17.144629
python sEMG_SVM.py --data ./dataset --test-size 0.2 --kernel rbf --grid
