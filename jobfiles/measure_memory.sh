#!/bin/bash



sparsity=(0.0 0.1 0.2 0.3 0.4 0.5 0.6)
baseline_configs=(
    "run_configs/evaluate/sparse-tetra/sparse-vit-boq.yaml"
    "run_configs/evaluate/sparse-tetra/sparse-vit-salad.yaml"
    "run_configs/evaluate/sparse-tetra/sparse-vit-mixvpr.yaml"
    "run_configs/evaluate/sparse-tetra/sparse-vit-cls.yaml"
)
for sparsity in "${sparsity[@]}"; do
    for baseline in "${baseline_configs[@]}"; do
        python measure_memory.py --config "$baseline" --sparsity "$sparsity"
    done
done



python measure_memory.py --config run_configs/evaluate/baselines/cosplace.yaml
python measure_memory.py --config run_configs/evaluate/baselines/dinoboq.yaml
python measure_memory.py --config run_configs/evaluate/baselines/dinosalad.yaml
python measure_memory.py --config run_configs/evaluate/baselines/eigenplace.yaml
python measure_memory.py --config run_configs/evaluate/baselines/mixvpr.yaml


