#!/usr/bin/env bash

python train.py --ontology cc --model_id 0
python predict.py --ontology cc --model_id 0
python evaluate.py --ontology cc --model_id 0