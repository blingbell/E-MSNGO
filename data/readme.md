# Data Readme

This is a description of the data folder.

**TestData**

Here you can quickly test the model by the testdata.

We provide less than 50 proteins, and then you can simply run preprocessing.sh and next training steps by setting {datapath} to data/testdata. This still takes several minutes.

* Structual feature is saved to testdata/sag_struct_feature_bp[/mf/cc].npy
* Sequence feature is saved to testdata/esm_feature.npy. If you don't want to download esm2_t33_650M_UR50D, you can use the esm_feature.npy we provide.


**Your Dataset**

If you want to use your own dataset, please follow the steps below:

* The PPI file is placed in the datadir/PPIdata directory, with a file name such as: 3702.protein.links.v11.0
* Download the ID mapping file from the string database to the uniprot database from [Uniprot](https://www.uniprot.org/), and put it in datadir/IDmap-uniprot-string, with the file name and format like uniprotkb_3702.xlsx
* Download all_uniprot_goa.gaf from [GOA website](https://www.ebi.ac.uk/GOA/), and put it into datadir/GO. We provide get_goa_spiece function in data_preprocessing.py to extract spieces goa you need, and you need to enable it in main function of data_preprocessing.py

Before you run preprocessing.sh, don't forget modify TAXIDS in utils.py

This may take a long time.


**Our Dataset**

We also provide the full dataset we used in the training, which can be downloaded from [here](https://drive.google.com/file/d/16J7P8FzmXdmYsv6dYnosA1-YCG5ngMWZ/view?usp=drive_link)
