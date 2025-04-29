#!/bin/bash

# List of config files and sparsity values
configs=(
    "run_configs/evaluate/sparse-tetra/sparse-vit-boq.yaml"
    "run_configs/evaluate/sparse-tetra/sparse-vit-salad.yaml" 
    "run_configs/evaluate/sparse-tetra/sparse-vit-mixvpr.yaml"
    "run_configs/evaluate/sparse-tetra/sparse-vit-cls.yaml"
)

sparsity_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)

# Loop through configs and sparsity values
for config in "${configs[@]}"; do
    for sparsity in "${sparsity_values[@]}"; do
        python evaluate.py --config "$config" --sparsity "$sparsity"
    done
done
