# MSNGO: Multi-species protein function annotation based on 3D protein structure and network propagation

This is the code repository for protein function prediction model MSNGO. 

**MSNGO** is a a multi-species protein function prediction model based on structural features and heterogeneous network propagation, which provides a structure encoder and can propagate structural feature on heterogeneous network for predicting Gene Ontology terms.

<div align=center><img width="800" alt="overview" src="https://github.com/blingbell/MSNGO/blob/master/images/overview.png"></div>

## Dependencies
* The code was developed and tested using python 3.8.
* To install python dependencies run: `pip install -r requirements.txt`. Some libraries may need to be installed via conda.
* The version of CUDA is `cudatoolkit==11.3.1`

## Data
<div align=center><img width="147" alt="uniprot" src="https://github.com/blingbell/MSNGO/blob/master/images/uniprot.jpg"><img width="175" alt="string" src="https://github.com/blingbell/MSNGO/blob/master/images/string.png"><img width="143" alt="goa" src="https://github.com/blingbell/MSNGO/blob/master/images/goa.png"><img width="206" alt="go" src="https://github.com/blingbell/MSNGO/blob/master/images/go.png"><img width="280" alt="alphafold" src="https://github.com/blingbell/MSNGO/blob/master/images/1724076793413.jpg"></div>

\
The data used are:
* Sequence: download from the [UniProt website](https://www.uniprot.org/).
* PPI Network: download from the [STRING website](https://string-db.org/).
* Annotation: download from the [GOA website](https://www.ebi.ac.uk/GOA/).
* Gene Ontology: download from the [GO website](http://geneontology.org/).
* ALphaFlod structure: download from the [AlphaFold website](https://alphafold.com/).

For a detailed description of data files, please see [here](data/readme.md).

## Predict
Run the `predict.py` script to make predictions about the input file (e.g. for MFO):
```
python predict.py --ontology mf -f your_test.fasta
```

## Train
preprocessing.sh is for processing your raw data. 

If you want to train on your own dataset, please read [here](data/readme.md). and download esm2_t33_650M_UR50D.pt to MSNGO/esm2_t33_650M_UR50D/

Then run the following command, but it may take a long time.
```
./scripts/preprocessing.sh
```

The mf, bp, and cc branches will be trained, predicted, and evaluated by the following files respectively.
```
./scripts/run_mf.sh
./scripts/run_bp.sh
./scripts/run_cc.sh
```

