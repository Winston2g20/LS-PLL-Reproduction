#!/bin/bash

if [ ! -d ".venv" ]; then 
    echo "Creating python environment..."
    python3 -m venv .venv

    echo "Installing dependencies..."
    ./.venv/bin/pip install -r requirements.txt
fi

./.venv/bin/python ./codes/main.py --model_path ./models --dataset_path ./datasets
