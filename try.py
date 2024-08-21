from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from io import StringIO
import requests as r
import numpy as np
from tqdm import tqdm
import os
from utils import get_uniprot2string
# newseqs = {}
# oldseqs = {}
# for record in SeqIO.parse(f'./data/network.fasta', 'fasta'):
#     newseqs[record.id] = record.seq
# for record in SeqIO.parse(f'/home/wangbb/mycode/dataset/network.fasta', 'fasta'):
#     oldseqs[record.id] = record.seq
# for pid in tqdm(newseqs):
#     if pid not in oldseqs:
#         print(f'{pid} not in old seqs')
#     if newseqs[pid]!= oldseqs[pid]:
#         print(f'{pid} is not the same')
#         print(f'old : {oldseqs[pid]}')
#         print(f'new : {newseqs[pid]}')


# dataset_seqs = []
# for record in SeqIO.parse(f'data/sequence/uniprotkb_71421.fasta', 'fasta'):
#     pid = str(record.id).split('|')[1]
#     seqs[pid] = record.seq
# print(type(record.id))
# pids = list(seqs.keys())
# print(type(seqs['P44579']))
# for pid in seqs:
#     dataset_seqs.append(SeqRecord(id=pid, seq=seqs[pid], description=''))
# # np.savetxt('data/sequence/71421_pids.txt',pids,fmt='%s')



# pid = 'E9Q180'
# baseUrl="http://www.uniprot.org/uniprot/"
# currentUrl=baseUrl+pid+".fasta"
# response = r.post(currentUrl)
# cData=''.join(response.text)
# Seq=StringIO(cData)
# print(response.text)
# result = list(SeqIO.parse(Seq,'fasta'))
# print(result)
# print(type(result[0].seq))
# seqs[pid]=SeqRecord(id=pid, seq=result[0].seq, description='')

# seqs = {}
# for tax in list(TAXIDS):
#     for record in SeqIO.parse(f'{datapath}/sequence/uniprotkb_{tax}.fasta', 'fasta'):
#         pid = str(record.id).split('|')[1]
#         seqs[pid] = record.seq


# for o in onts:
#     for c in cate:
#         other_pids = []
#         for pid in tqdm(data_set[o][c], desc= 'download Sequences...'):
#             if pid not in seqs:
#                 baseUrl="http://www.uniprot.org/uniprot/"
#                 currentUrl=baseUrl+pid+".fasta"
#                 response = r.post(currentUrl)
#                 cData=''.join(response.text)
#                 Seq=StringIO(cData)
#                 result = list(SeqIO.parse(Seq,'fasta'))
#                 if len(result)==0:
#                     print(pid)
#                 seqs[pid]=result[0].seq


pids = []
files = [file for file in os.listdir('/home/wangbb/MSNGO/data/structdata') if file.endswith('.pdb')]
print(files[0])
for file in files:
    if  os.path.getsize(f'/home/wangbb/MSNGO/data/structdata/{file}') < 100:
        pids.append(file)
np.savetxt('pdb_empty.txt', pids, fmt='%s')


# uniprot2string = get_uniprot2string('data')
# seqs = {}
# for record in SeqIO.parse(f'./data/network.fasta', 'fasta'):
#     seqs[record.id] = record.seq
# # files = np.loadtxt('pdb_empty.txt', dtype=str).tolist()
# files = ['AF-D4AB70-F1-model_v4.pdb']
# for file in files:
#     pid = uniprot2string[file.split('-')[1]]
#     os.system(f'curl -X POST --data "{str(seqs[pid]).replace("U", "X")[:400]}" https://api.esmatlas.com/foldSequence/v1/pdb/ > data/structdata/{file} -k')
