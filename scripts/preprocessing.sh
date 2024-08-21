#!/usr/bin/env bash

python data_preprocessing.py
python seq_embedding.py -m ./esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt
python struct2graph.py
python struct_embedding.py -ont mf -m ./structmodel
python struct_embedding.py -ont bp -m ./structmodel
python struct_embedding.py -ont cc -m ./structmodel

