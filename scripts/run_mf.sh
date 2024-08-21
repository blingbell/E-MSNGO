#!/usr/bin/env bash

python train.py --ontology mf --model_id 0
python predict.py --ontology mf --model_id 0
python evaluate.py --ontology mf --model_id 0
