import argparse
from logzero import logger
import numpy as np
import pandas as pd 
from tqdm import tqdm
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import networkx as nx
import random
import dgl
import torch
import os
from utils import get_uniprot2string, get_ppi_pid2index, get_pid_ont
import warnings

class Graph():
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(1.0/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(1.0/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)
    
    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [1.0 for nbr in range(len(list(G.neighbors(node))))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
        
        return 
    

def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def add_onehot(vec, one):
    if vec.shape[0] != one.shape[0]:
        if vec.shape[0] < one.shape[0]:
            one = one[:vec.shape[0],:]
        else:
            one = np.vstack((one,np.zeros((vec.shape[0]-one.shape[0], one.shape[1]))))

    node_vec=np.hstack((vec,one))
    return node_vec

def learn_embeddings(walks, walk_file, nums_node):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    
    with open(walk_file, 'w') as f:
        for ws in walks:
            line = ''
            for w in ws:
                line += str(w) + ' '
            line += '\n'
            f.write(line) 
    walks = LineSentence(walk_file)
    model = Word2Vec(walks, vector_size=30, window=5, min_count=0, hs=1, sg=1, workers=32, epochs=3)
    os.remove(walk_file)
    vectors = []
    for i in range(nums_node):
        if i in model.wv.key_to_index:
            vectors.append(list(model.wv[str(i)]))
        else:
            vectors.append(list(np.zeros(30)))

    return np.array(vectors)

def read_graph(filename):
    '''
    Reads the input network in networkx.
    '''
    G = nx.read_edgelist(filename, nodetype=int, data=False, create_using=nx.Graph())
    #G = G.to_undirected()

    return G

def _load_cmap(filename, cmap_thresh=10.0):
        
        D = load_predicted_PDB(filename)
        A = np.double(D < cmap_thresh)
        #print(A)
        A = A.reshape(1, *A.shape)

        return A


def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    warnings.simplefilter("ignore")
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    #name = structure.header['name']
    #print(name)
    residues = [r for r in structure.get_residues()]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one-two)

    return distances


def get_structmap(path, file_name, map_path):
    A = _load_cmap(os.path.join(path, file_name),cmap_thresh=10.0)
    B = np.reshape(A,(-1,len(A[0])))
    result = []
    N = len(B)
    for i in range(N):
        for j in range(N):
            tmp1 = []
            if B[i][j] and i!=j:
                tmp1.append(i)
                tmp1.append(j)
                result.append(tmp1)
    np.array(result)
    #print(result)
    data = pd.DataFrame(result)

    if '-' in file_name:
        name = file_name.split('-')[1]
    else:
        name = file_name.split('.')[0]
    data.to_csv(f"{map_path}/{name}.txt",sep=" ",index=False, header=False)
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')
    parser.add_argument("-d", "--datapath", type=str, default='data', help="path to save data")
    parser.add_argument("-p", "--mapdir", type=str, default='structmap', help="path to save contact map data")
    
    args = parser.parse_args()
    logger.info(args)
    datapath = args.datapath
    outpath = f'{datapath}/{args.mapdir}'
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        
    uniprot2string = get_uniprot2string(datapath)
    ppi_pid2index = get_ppi_pid2index(datapath)

    # convert raw structural data into contact maps
    # file_name = "AF-A0A0A7EPL0-F1-model_v4.pdb"
    files = [file for file in os.listdir(f'{datapath}/structdata')]
    map_names = [file.split('.')[0] for file in os.listdir(outpath)]
    for file in tqdm(files, desc=f"struct to map"):
        if file.endswith(".pdb"):
            name = file.split("-")[1]
            if name in map_names:
                continue
            if name in uniprot2string and uniprot2string[name] in ppi_pid2index:
                get_structmap(f'{datapath}/structdata', file, outpath)
                map_names.append(name)
                # get_structmap(f'data/structdata', file, outpath)

            ## If the storage space is insufficient, you can uncomment the following code to delete the original structure data
            # try:
            #     os.remove(os.path.join(f'{datapath}/structdata', file))
            # except OSError as e:
            #     print(f"Error in struct2map {file} : can not remove it . {e}")
    
    # Perform node2vec on contact maps and save them to dgl graphs
    seqs = {}
    for record in SeqIO.parse(f'{datapath}/network.fasta', 'fasta'):
        seqs[str(record.id)] = str(record.seq)

    directed = False
    p = 0.8
    q = 1.2
    num_walks = 5 # every node walks 5 times
    walk_length = 30
    emb_graph_mf = {}
    emb_graph_cc = {}
    emb_graph_bp = {}

    bp_pids = get_pid_ont('bp', datapath)
    mf_pids = get_pid_ont('mf', datapath)
    cc_pids = get_pid_ont('cc', datapath)

    map_files = [file for file in os.listdir(outpath) if file.endswith('.txt')]
    for edgefile in tqdm(map_files, desc = 'embedding node2vec......'):
        pid = edgefile.split('.')[0]
        
        nx_G = read_graph(os.path.join(outpath, edgefile))
        seq = seqs[uniprot2string[pid]]
        G = Graph(nx_G, directed, p, q)
        G.preprocess_transition_probs()

        # random walk and node2vec
        walks = G.simulate_walks(num_walks, walk_length)
        walk_file = os.path.join(outpath,  pid + '_walks.txt')
        vectors = learn_embeddings(walks, walk_file, nx_G.number_of_nodes())
        # add onehot information to node feature
        onehot = seq2onehot(seq)
        protein_nvec = add_onehot(vectors, onehot)
        # store graph and node feature to dgl graph
        g = dgl.from_networkx(nx_G)
        g = dgl.add_self_loop(g)
        g.ndata['feature'] = torch.tensor(protein_nvec, dtype=torch.float32)
        if pid in bp_pids:
            emb_graph_bp[pid] = g
        if pid in mf_pids:
            emb_graph_mf[pid] = g
        if pid in cc_pids:
            emb_graph_cc[pid] = g

    np.save(f'{datapath}/emb_graph_cc.npy', emb_graph_cc)  
    np.save(f'{datapath}/emb_graph_mf.npy', emb_graph_mf)
    np.save(f'{datapath}/emb_graph_bp.npy', emb_graph_bp)

    logger.info('success to save emb_graph')

    