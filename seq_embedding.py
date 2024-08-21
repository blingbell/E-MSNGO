import argparse
from logzero import logger
import shutil
import torch
import esm
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import os
from utils import get_uniprot2string, get_ppi_pid2index
"""
esm-extract esm2_t33_650M_UR50D examples/data/some_proteins.fasta \
  examples/data/some_proteins_emb_esm2 --repr_layers 0 32 33 --include

@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
"""
# Load ESM-2 model
def get_seq_feature(datapath:str, modelfile:str):

    model, alphabet = esm.pretrained.load_model_and_alphabet(modelfile)
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    if not os.path.exists(f'{datapath}/seq_feature'):
        os.mkdir(f'{datapath}/seq_feature')

    files_done = [file.split('.')[0] for file in os.listdir(f'{datapath}/seq_feature')]
    seqs_done = set(files_done)
    files = [file for file in os.listdir(f'{datapath}/split_seqs')]
    for file in files:
        logger.info(f'{file} to be processing')
        seqs = list()
        for record in SeqIO.parse(f'{datapath}/split_seqs/{file}', 'fasta'):
            if str(record.id) in seqs_done:
                continue
            seqs.append((str(record.id), str(record.seq)))
        if len(seqs) == 0:
            continue

        for pid, data in tqdm(seqs, desc = 'esm2 embedding...'):
            batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(pid, data)))
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]

            # for i, tokens_len in enumerate(batch_lens):
            np.savetxt(f'{datapath}/seq_feature/{pid}.txt',token_representations[0, 1 : batch_lens[0] - 1].mean(0).numpy().tolist())
            seqs_done.add(pid)
        os.remove(f'{datapath}/split_seqs/{file}')
    try:
        os.rmdir(f'{datapath}/split_seqs')
    except Exception as e:
        print(f'Error! Some seqs are not embeded in {datapath}/split_seqs, please check!!')
    return 

def split_network_seqs(datapath:str):
    string2uniprot = {}
    uniprot2string = get_uniprot2string(datapath)
    for uniprotid in uniprot2string:
        string2uniprot[uniprot2string[uniprotid]] = uniprotid

    seqs = dict()
    for record in SeqIO.parse(f'{datapath}/network.fasta', 'fasta'):
        seqs[string2uniprot[str(record.id)]] = str(record.seq)

    # split seqs to small sets
    length = 250
    
    split_seqs = dict()
    for i in range(len(seqs)//length+1):
        split_seqs[str(i)] = []
    index = 0
    for pid in seqs:
        split_seqs[str(index//length)].append( SeqRecord(id=pid, seq=seqs[pid], description='') )
        index += 1

    if not os.path.exists(f'{datapath}/split_seqs'):
        os.mkdir(f'{datapath}/split_seqs')

    for i in split_seqs:
        ct = SeqIO.write(split_seqs[i], f'{datapath}/split_seqs/network_{i}.fasta', 'fasta')
    return 

def get_seq_embedding(datapath:str):
    ppi_pid2index = get_ppi_pid2index(datapath)
    uniprot2string = get_uniprot2string(datapath)
    feature_matrix = np.zeros((len(ppi_pid2index), 1280))
    
    pids = [file.split('.')[0] for file in os.listdir(f'{datapath}/seq_feature/') if file.endswith('.txt')]
    assert len(pids) == len(ppi_pid2index)
    for pid in tqdm(pids, desc = 'get seq embedding...'):
        try :
            feature_matrix[int(ppi_pid2index[uniprot2string[pid]])] = np.loadtxt(f'{datapath}/seq_feature/{pid}.txt').astype(np.float32)
        except Exception as e:
            print(f'Pack {pid} seq embedding  Error! {e}')
    np.save(f'{datapath}/esm_feature.npy', feature_matrix)

    logger.info('Seq embedding finish!')

    try:
        shutil.rmtree(f'{datapath}/seq_feature')
    except Exception as e:
        print(f"seq_feature Folder does not exist!  {e}")
    return 




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument("-d", "--datapath", type=str, default='data',  
            help="path to save data")
    parser.add_argument("-m", "--modelfile", type=str, default='./esm2_t33_650M_UR50D/esm2_t33_650M_UR50D.pt',
            help="The path of ESM-2 model .pt file")
    
    args = parser.parse_args()
    logger.info(args)
    # get seq embedding
    split_network_seqs(args.datapath)
    get_seq_feature(args.datapath, args.modelfile)
    get_seq_embedding(args.datapath)

