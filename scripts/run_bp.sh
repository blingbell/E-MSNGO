#!/usr/bin/env bash

python train.py --ontology bp --model_id 0
python predict.py --ontology bp --model_id 0
python evaluate.py --ontology bp --model_id 0
