#!/bin/bash

if [ ! -d ".venv" ]; then 
    echo "Creating python environment..."
    python3 -m venv .venv

    echo "Installing dependencies..."
    ./.venv/bin/pip install -r requirements.txt
fi

LOG_FILE="logs/$(date +"%Y%m%d_%H%M%S").log"
mkdir -p logs

./.venv/bin/python -u ./codes/main.py --model_path ./models --dataset_path ./datasets 2>&1 | tee "$LOG_FILE"
