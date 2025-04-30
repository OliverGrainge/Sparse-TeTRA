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
#for config in "${configs[@]}"; do
#    for sparsity in "${sparsity_values[@]}"; do
#        python evaluate.py --config "$config" --sparsity "$sparsity"
#    done
#done



baseline_configs=(
    "run_configs/evaluate/baselines/dinoboq.yaml"
    "run_configs/evaluate/baselines/cosplace.yaml"
    "run_configs/evaluate/baselines/eigenplace.yaml"
    "run_configs/evaluate/baselines/dinosalad.yaml"
    "run_configs/evaluate/baselines/mixvpr.yaml"
)

# Run evaluation for each baseline config
for baseline_config in "${baseline_configs[@]}"; do
    python evaluate.py --config "$baseline_config"
done






