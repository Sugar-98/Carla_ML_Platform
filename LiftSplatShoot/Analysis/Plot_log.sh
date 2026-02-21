#!/bin/bash
# Analyze log.csv in the current directory with plot_train_log.py and output results to ./log_plot

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python3 "${SCRIPT_DIR}/plot_train_log.py" \
    --csv ./log.csv \
    --outdir ./log_plot
