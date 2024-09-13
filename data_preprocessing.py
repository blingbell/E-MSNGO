
import argparse
from collections import defaultdict
import requests as r
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
import dgl
import dgl.data
import torch
from Bio import SeqIO
from io import StringIO
from Bio.SeqRecord import SeqRecord
from logzero import logger
from scipy.sparse import *
from scipy import *
import numpy as np
import json
import wget
from ontology import GeneOntology
from utils import *

func_dict = {
    'C': 'cc',
    'P': 'bp',
    'F': 'mf'
}
ROOT_GO_TERMS = {'GO:0003674', 'GO:0008150', 'GO:0005575'}

EXP_CODES = set(['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP', 'IBA', 'IBD', 'IKR', 'IRD'])

# TARGETS = set([
#     '3702', '4577', '284812', '559292', '6239', '7227', '7955', '9606', 
#     '10090', '10116', '44689', '83333', '71421'
# ])

def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    version: int = 4
    ):
    
    BASE_URL = "https://alphafold.ebi.ac.uk/files/"
    uniprot_id = uniprot_id.upper()

    query_url = f"{BASE_URL}AF-{uniprot_id}-F1-model_v{version}.pdb"
    structure_filename = os.path.join(out_dir, f"AF-{uniprot_id}-F1-model_v{version}.pdb")
    if os.path.exists(structure_filename):
        return structure_filename
    try:
        structure_filename = wget.download(query_url, out=out_dir)
    except:
        return None
    
    return structure_filename

def get_goa_spiece(datapath:str):
    all_data = {}
    spieces = TAXIDS
    if 'testdata' in datapath:
        spieces = TAXIDS_test
    for id in spieces:
        all_data[id]=[]
    f = open(f"{datapath}/GO/goa_uniprot_all.gaf","r")
    line = f.readline()
    while line.startswith('!'):
        line = f.readline()

    while line != '':
        data = line.strip().split('\t')
        taxon = data[12].strip().split(':')[1]
        eid = data[6]
        if taxon in spieces and eid in EXP_CODES:
            all_data[taxon].append({'PID':f'{data[1]}', 'GO':f'{data[4]}', 'EID':f'{eid}', 'taxon':f'{taxon}', 'aspect':f'{data[8]}', 'time':f'{data[13]}'})
            # print(f'PID:{data[1]}, GO:{data[4]}, EID:{eid}, taxon:{taxon}, aspect:{data[8]}, time:{data[13]}')
        line = f.readline()
    f.close()

    # output
    for id in all_data:
        with open(f'{datapath}/GO/goa_{id}.json','w') as f:
            for item in all_data[id]:
                jsonitem = json.dumps(item)
                f.write(jsonitem)
                f.write('\n')
    return

def get_dataset(datapath:str):
    filename = f'{datapath}/GO/goa'
    go_file = f'{datapath}/GO/go.obo'

    spieces = TAXIDS
    if 'testdata' in datapath:
        spieces = TAXIDS_test
        
    onts = ['bp', 'mf', 'cc']
    cate = ['train', 'valid', 'test']
    data_set = dict()
    for o in onts:
        data_set[o] = dict()
        if not os.path.exists(f'{datapath}/{o}'):
            os.mkdir(f'{datapath}/{o}')
        for c in cate:
            data_set[o][c] = defaultdict(set)

    data_size_orgs = dict()  # statistic dataset size (train, valid, test)
    for i in list(spieces):
        data_size_orgs[i] = dict()
        for o in onts:
            data_size_orgs[i][o] = dict()
            for c in cate:
                data_size_orgs[i][o][c] = set()

    uniprot2string = get_uniprot2string(datapath)
    prot_annots = defaultdict(dict)
    # get goa annotations
    # one example line of the file： 
    # {"PID": "P54144", "GO": "GO:0043621", "EID": "IDA", "taxon": "3702", "aspect": "F", "time": "20170329"}
    
    for target in spieces:
        with open(f'{filename}_{target}.json', 'r') as f:
            for line in tqdm(f.readlines(), desc=f'Reading {target} annotations......'):
                line = json.loads(line)
                pid = line['PID']
                annot = line['GO']
                code = line['EID']
                ont = func_dict[line['aspect']]
                org_id = line['taxon']
                date = line['time']
                if code not in EXP_CODES:
                    continue
                if uniprot2string.get(pid) == None: # uniprot  string
                    continue
                if date >= '20230801': # only proteins annotated before 20230801
                    continue
                if prot_annots.get(pid) == None:
                    prot_annots[pid]['annotation'] = set()
                    prot_annots[pid]['annotation'].add( (annot, ont) )
                    prot_annots[pid]['org_id'] = org_id
                    prot_annots[pid]['min_date'] = date
                else:
                    prot_annots[pid]['annotation'].add( (annot, ont) )
                    prot_annots[pid]['min_date'] = min(prot_annots[pid]['min_date'], date) #store first date
        f.close()
    # Propagate annotations
    go = GeneOntology(go_file)
    labels_list = list()
    for pid in prot_annots:
        annot_set = set()
        for annot, _ in prot_annots[pid]['annotation']:
            annot_set |= go.get_ancestors(annot) 
        annot_set -= ROOT_GO_TERMS
        labels_list.append(list(annot_set))
        annots = set()
        for annot in annot_set:
            annots.add( (annot, go.get_namespace(annot)) )
        prot_annots[pid]['annotation'] = annots # update

    go.calculate_ic(labels_list)
    logger.info('size of target protein: ' + str(len(prot_annots)))

    # split data set according to time
    for pid, item in tqdm(prot_annots.items(), desc='split data set.......', total=len(prot_annots)):
        min_date = item['min_date']
        org_id = item['org_id']
        annots = item['annotation']
        if min_date >= '20220801' and min_date < '20230801':
            for annot, o in annots:
                data_size_orgs[org_id][o]['test'].add(pid)
                data_set[o]['test'][pid].add(annot)
        elif min_date >= '20210101' and min_date < '20220801':
            for annot, o in annots:
                data_size_orgs[org_id][o]['valid'].add(pid)
                data_set[o]['valid'][pid].add(annot)
        elif min_date < '20210101':
            for annot, o in annots:
                data_size_orgs[org_id][o]['train'].add(pid)
                data_set[o]['train'][pid].add(annot)


    # preprocess protein sequence, save to .fasta
    seqs = {}
    for tax in list(spieces):
        if os.path.exists(f'{datapath}/sequence/uniprotkb_{tax}.fasta'):
            for record in SeqIO.parse(f'{datapath}/sequence/uniprotkb_{tax}.fasta', 'fasta'):
                pid = str(record.id).split('|')[1]
                seqs[pid] = record.seq

    logger.info('preprocess seqs.')

    for o in onts:
        for c in cate:
            for pid in tqdm(data_set[o][c], desc= 'download Sequences...'):
                if pid not in seqs:
                    baseUrl="http://www.uniprot.org/uniprot/"
                    currentUrl=baseUrl+pid+".fasta"
                    response = r.post(currentUrl)
                    cData=''.join(response.text)
                    Seq=StringIO(cData)
                    result = list(SeqIO.parse(Seq,'fasta'))
                    seqs[pid]=result[0].seq
    for o in onts:
        for c in cate:
            dataset_seqs = []
            dataset_pids = []
            for pid in tqdm(data_set[o][c], desc= 'split Sequences to dataset...'):
                dataset_seqs.append(SeqRecord(id=pid, seq=seqs[pid], description=''))
                dataset_pids.append(pid)
            ct = SeqIO.write(dataset_seqs, f'{datapath}/{o}/{o}_{c}.fasta', 'fasta')
            np.savetxt(f'{datapath}/{o}/{o}_{c}_pids.txt', dataset_pids, fmt = '%s')

    # all used sequence are saved to network.fasta
    logger.info('get network.fasta')
    network_seqs = dict()
    index = 0
    ppi_pid2index = dict()
    for o in onts:
        for c in cate:
            for pid in data_set[o][c]:
                if pid not in network_seqs:
                    network_seqs[pid]=SeqRecord(id=uniprot2string[pid], seq=seqs[pid], description='')
                    ppi_pid2index[uniprot2string[pid]] = index
                    index+=1
    all_seqs_ct = SeqIO.write(list(network_seqs.values()), f'{datapath}/network.fasta', 'fasta')
    
    with open(f'{datapath}/ppi_pid2index.txt', 'w') as f:
        for pid in ppi_pid2index:
            f.write(f'{pid} {ppi_pid2index[pid]}\n')
    print(f'len ppi_pid2index = {len(ppi_pid2index)}')
    assert len(ppi_pid2index) == all_seqs_ct

    # download structure
    logger.info('get raw structure')
    if not os.path.exists(f'{datapath}/structdata'):
        os.mkdir(f'{datapath}/structdata')
    for pid in tqdm(network_seqs, desc = 'download Structures'):
        flag = download_alphafold_structure(pid, f'{datapath}/structdata')
        if flag == None or os.path.getsize(f'{datapath}/structdata/AF-{pid.upper()}-F1-model_v4.pdb') < 100:
            # esm-fold predict structure
            os.system(f'curl -X POST --data "{str(network_seqs[pid].seq).replace("U", "X")[:400]}" https://api.esmatlas.com/foldSequence/v1/pdb/ > {datapath}/structdata/AF-{str(pid).upper()}-F1-model_v4.pdb -k')

    logger.info('save goa data to _go.txt')
    for o in onts:
        for c in cate:
            with open(f'{datapath}/{o}/{o}_{c}_go.txt', 'w') as f:
                for pid in data_set[o][c]:
                    for go_id in data_set[o][c][pid]:
                        f.write(f"{pid}\t{go_id}\t{o}\t{prot_annots[pid]['org_id']}\n")
            f.close()
    logger.info('get go ic.')
    for o in onts:
        ont_go_set = set()
        for c in cate:
            for pid in data_set[o][c]:
                ont_go_set |= data_set[o][c][pid]
        sorted_go_list = list()
        for go_id in go.term_top_sort(o):
            if go_id in ont_go_set:
                sorted_go_list.append(go_id)
        assert len(ont_go_set) == len(sorted_go_list)
        with open(f'{datapath}/{o}/{o}_go_ic.txt', 'w') as f:
            for go_id in sorted_go_list:
                f.write(f'{go_id}\t{go.get_ic(go_id)}\n')
        f.close()
    
    with open(f'{datapath}/data_set_size.txt', 'w') as f:
        for o in onts:
            f.write(o+'\n')
            for c in cate:
                f.write('\t' + c + ' set: ' + str( len(data_set[o][c]) ) +'\n')
        f.write('\n')
        for i in list(spieces):
            f.write(str(i)+'\n')
            for o in onts:
                f.write('\t' + str(o) + '\n')
                for c in cate:
                    f.write('\t\t' + str(c)+ ': ' + str(len(data_size_orgs[i][o][c])) + '\n')
    return 

def get_norm_mat(net_mat: csr_matrix)-> csr_matrix:
    """
    Normalization: Construct two diagonal matrices by summing in the row and column directions respectively, 
    then multiply these two diagonal matrices with the original matrix net_mat, and the result is the return value. 
    """
    colsum = np.array(net_mat.sum(0))
    mat_d_0 = np.power(colsum, -0.5).flatten()
    mat_d_0[np.isinf(mat_d_0)] = 0.
    mat_d_0 = diags(mat_d_0, format='csr')
    rowsum = np.array(net_mat.sum(1))
    mat_d_1 = np.power(rowsum, -0.5).flatten()
    mat_d_1[np.isinf(mat_d_1)] = 0.
    mat_d_1 = diags(mat_d_1, format='csr')
    return mat_d_0 @ net_mat @ mat_d_1

def get_sim_mat(datapath:str, e_value):
    make_diamond_db(f'{datapath}/network.fasta', f'{datapath}/network_db.dmnd')
    diamond_homo(f'{datapath}/network_db.dmnd', f'{datapath}/network.fasta', f'{datapath}/network_sim_diamond.txt')
    logger.info('network diamond db and network_sim have completed!')
    u_list = []
    v_list = []
    weight = []
    ppi_pid2index = get_ppi_pid2index(datapath)
    with open(f'{datapath}/network_sim_diamond.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            record = line.strip().split('\t')
            if float(record[2]) < e_value: # e-value = 1e-04
                if ppi_pid2index.get(record[0]) != None and ppi_pid2index.get(record[1]) != None:
                    u_list.append(int(ppi_pid2index[record[0]]))
                    v_list.append(int(ppi_pid2index[record[1]]))
                    weight.append(float(record[3]))
    assert len(u_list) == len(v_list) and len(v_list) == len(weight)
    mat_size = len(ppi_pid2index)
    logger.info(f'building sim_matrix. ')
    logger.info(f'matrix size = ({mat_size}, {mat_size})')
    sim_mat = np.zeros((mat_size, mat_size), dtype = float)
    for u, v, w in tqdm(zip(u_list, v_list, weight), desc='Building sim matrix......'):
        sim_mat[u][v] = w
    sim_csr_mat = csr_matrix(sim_mat)
    save_npz(f'{datapath}/sim_matrix.npz', sim_csr_mat)
    return 
    

def get_ppi_mat(datapath:str):
    ppi_pid2index = get_ppi_pid2index(datapath)
    logger.info(f'reading ppi_links file. ')
    u_list = []
    v_list = []
    weight = []
    spieces = TAXIDS
    if 'testdata' in datapath:
        spieces = TAXIDS_test
    for tax in tqdm(spieces, desc='Reading all spieces ppi link file......'):
        with open(f'{datapath}/PPIdata/{tax}.protein.links.v11.0.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(' ')
                if ppi_pid2index.get(items[0]) != None and ppi_pid2index.get(items[1]) != None:
                    u_list.append(int(ppi_pid2index[items[0]]))
                    v_list.append(int(ppi_pid2index[items[1]]))
                    weight.append(float(items[2]))
        f.close()
    assert len(u_list) == len(v_list) and len(v_list) == len(weight)

    mat_size = len(ppi_pid2index)
    logger.info(f'building ppi_matrix. ')
    print(f'matrix size = ({mat_size}, {mat_size})')
    ppi_mat = np.zeros((mat_size, mat_size), dtype = float)
    for u, v, w in tqdm(zip(u_list, v_list, weight), desc='Building ppi matrix......'):
        ppi_mat[u][v] = w
    ppi_csr_mat = csr_matrix(ppi_mat)
    save_npz(f'{datapath}/ppi_matrix.npz', ppi_csr_mat)
    return 


def build_hetero_network(ppi_top: int, datapath:str):
    sim_mat_path = f'{datapath}/sim_matrix.npz'
    ppi_mat_path = f'{datapath}/ppi_matrix.npz'
    # load matrix
    sim_csr_mat = sparse.load_npz(sim_mat_path)
    ppi_csr_mat = sparse.load_npz(ppi_mat_path)
    logger.info(F'sim matrix shape: {sim_csr_mat.shape}, ppi matrix shape: {ppi_csr_mat.shape}')

    # get similarity edges
    sim_u = list()
    sim_v = list()
    sim_weight = list()
    for u in trange(sim_csr_mat.shape[0]):
        for w, v in zip(sim_csr_mat[u].data, sim_csr_mat[u].indices):
            sim_u.append(u)
            sim_v.append(v)
            sim_weight.append(w)
    if 'testdata' in datapath:
        sim_norm_mat = csr_matrix((sim_weight, (sim_u, sim_v)), shape=sim_csr_mat.shape).tocoo()
    else:
        sim_norm_mat = get_norm_mat(csr_matrix((sim_weight, (sim_u, sim_v)), shape=sim_csr_mat.shape)).tocoo()
    sim_edge_u = list()
    sim_edge_v = list()
    sim_edge_weight = list()
    for u, v, w in tqdm(zip(sim_norm_mat.row, sim_norm_mat.col, sim_norm_mat.data)):
            sim_edge_u.append(v)
            sim_edge_v.append(u)
            sim_edge_weight.append(w)
        
    # get interaction top edges
    ppi_top_u = list()
    ppi_top_v = list()
    ppi_top_w = list()
    for u in trange(ppi_csr_mat.shape[0]):
        for w, v in sorted(zip(ppi_csr_mat[u].data, ppi_csr_mat[u].indices), reverse=True)[:ppi_top]:
            ppi_top_u.append(u)
            ppi_top_v.append(v)
            ppi_top_w.append(w)
    if 'testdata' in datapath:
        ppi_norm_mat = csr_matrix((ppi_top_w, (ppi_top_u, ppi_top_v)), shape=ppi_csr_mat.shape).tocoo()
    else:        
        ppi_norm_mat = get_norm_mat(csr_matrix((ppi_top_w, (ppi_top_u, ppi_top_v)), shape=ppi_csr_mat.shape)).tocoo()
    ppi_edge_u = list()
    ppi_edge_v = list()
    ppi_edge_weight = list()
    for u, v, w in tqdm(zip(ppi_norm_mat.row, ppi_norm_mat.col, ppi_norm_mat.data)):
            ppi_edge_u.append(v)
            ppi_edge_v.append(u)
            ppi_edge_weight.append(w)
    # create and save heterograph
            
    ppsi_g = dgl.heterograph({
        ('protein', 'sim', 'protein'): (torch.tensor(sim_edge_u), torch.tensor(sim_edge_v)),
        ('protein', 'ppi', 'protein'): (torch.tensor(ppi_edge_u), torch.tensor(ppi_edge_v))
    })
    ppsi_g.edges['sim'].data['w'] = torch.tensor(sim_edge_weight).float()
    ppsi_g.edges['ppi'].data['w'] = torch.tensor(ppi_edge_weight).float()
    logger.info(F"heterograph info: {ppsi_g}")
    #assert ppsi_g.in_degrees(etype='sim').max() <= sim
    assert ppsi_g.in_degrees(etype='ppi').max() <= ppi_top
    dgl.data.utils.save_graphs(F'{datapath}/dgl_hetero', ppsi_g)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument("-d", "--datapath", type=str, default='data',  help="path to save data")
    parser.add_argument("--ppi_k", type=int, default=50, help="selecting interaction edges with k largest weights for each node")
    parser.add_argument("--e_value", type=float, default=1e-04,  help="Diamond hyperparameters")
    
    args = parser.parse_args()
    logger.info(args)
    datapath = args.datapath
    # get goa, sequence and structure data and split them to dataset 
    # Download structure data may take a long time
    get_dataset(datapath)
    # get ppi net and sequence similarity net
    get_sim_mat(datapath, args.e_value)
    get_ppi_mat(datapath)
    # get heterogeneous network
    build_hetero_network(args.ppi_k, datapath)
    
