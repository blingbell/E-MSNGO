# Data Readme

This is a description of the data folder.

**TestData**

Here you can quickly test the model by the testdata.

We provide less than 50 proteins, and then you can simply run preprocessing.sh and training steps by setting {datapath} to testdata. This still takes several minutes.

* Structual feature is saved to testdata/emb_graph_bp[/mf/cc].npy
* Sequence feature is saved to testdata/esm_feature.npy
If you don't want to download esm2_t33_650M_UR50D, you can use 

**Your Dataset**

If you want to use your own dataset, please follow the steps below:

* The PPI file is placed in the data/PPIdata directory, with a file name such as: 3702.protein.links.v11.0
* Download the ID mapping file from the string database to the uniprot database from [Uniprot](https://www.uniprot.org/), and put it in data/IDmap-uniprot-string, with the file name and format like uniprotkb_3702.xlsx

Then you can run preprocessing.sh. Don't forget modify TAXIDS in utils.py

