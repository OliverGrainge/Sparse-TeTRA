#!/bin/bash

python posttrain.py --config run_configs/posttrain/boq.yaml
python posttrain.py --config run_configs/posttrain/salad.yaml
python posttrain.py --config run_configs/posttrain/mixvpr.yaml
python posttrain.py --config run_configs/posttrain/cls.yaml
