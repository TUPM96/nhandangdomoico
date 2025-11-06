#!/usr/bin/env bash
set -euo pipefail
# Re-run command captured at 2025-11-07T02:03:11.466775
python sEMG_SVM.py --data ./dataset --test-size 0.2 --kernel rbf --grid
