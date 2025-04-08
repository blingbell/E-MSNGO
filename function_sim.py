import numpy as np
import argparse
from utils import get_go_ic, get_ppi_pid2index, get_uniprot2string, get_tax_pid_label_dict, TAXIDS, TAXIDS_test
from logzero import logger
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import dgl
import torch


# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False 


def resnik_similarity(go1, go2, resnik, ic_values:dict):
    ancestors1 = resnik.get_ancestors(go1)
    ancestors2 = resnik.get_ancestors(go2)
    common_ancestors = ancestors1 & ancestors2

    if not common_ancestors:
        return 0.0  

    return max(ic_values.get(anc, 0.0) for anc in common_ancestors)


def compute_cross_resnik_similarity(ppi1_labels, ppi2_labels, ic_values:dict):
    """
    Calculate the Resnik similarity of protein pairs
    :param go_annotations1: dict, GO terms of proteins in PPI1
    :param go_annotations2: dict, GO terms of proteins in PPI2
    :param resnik: Resnik similarity calculator
    :return: dict, similarity of protein pairs
    """
    resnik_sim = {}

    for p1, go_set1 in ppi1_labels.items():
        for p2, go_set2 in ppi2_labels.items():
            common_ancestors = go_set1 & go_set2
            if not common_ancestors:
                max_resnik = 0.0
            else:
                max_resnik = max(ic_values.get(anc, 0.0) for anc in common_ancestors)
            if max_resnik > 0:
                resnik_sim[(p1, p2)] = max_resnik
    
    scores_array = np.array(list(resnik_sim.values()))
    beta = np.median(scores_array)  
    alpha = 1.0
    return {pair: 1 / (1 + np.exp(-alpha * (score - beta))) for pair, score in resnik_sim.items()}


def compute_cross_jaccard_similarity(ppi1_labels, ppi2_labels):
    """
    Calculate the Jaccard similarity between all protein pairs across two PPI networks.

    :param ppi1_labels: dict, sets of functional labels for proteins in PPI1
    :param ppi2_labels: dict, sets of functional labels for proteins in PPI2
    :return: dict, Jaccard similarity matrix (only storing protein pairs with similarity > 0)
    """
    jaccard_sim = {}

    for p1, labels1 in ppi1_labels.items():
        for p2, labels2 in ppi2_labels.items():
            intersection = len(labels1 & labels2)
            union = len(labels1 | labels2)
            
            if union > 0:
                sim = intersection / union
                if sim > 0:  
                    jaccard_sim[(p1, p2)] = sim

    return jaccard_sim


def weighted_sum(jaccard_sim, resnik_sim, alpha=0.5):
    pair_sim = {}

    for pair, value_j in jaccard_sim.items():
        value_r = resnik_sim.get(pair, 0.0)
        value = alpha*value_j + (1-alpha)*value_r
        pair_sim[pair] = value

    return pair_sim


def add_ppi_edges(graph, edge_dict):

    src_nodes, dst_nodes = zip(*edge_dict.keys())
    edge_weights = list(edge_dict.values())

    src_tensor = torch.tensor(src_nodes, dtype=torch.int32)
    dst_tensor = torch.tensor(dst_nodes, dtype=torch.int32)
    edge_weight_tensor = torch.tensor(edge_weights).float()

    # match weight
    existing_weight = graph.edges['ppi'].data['w'] if 'w' in graph.edges['ppi'].data else torch.tensor([]).float()
    new_weight = torch.cat((existing_weight, edge_weight_tensor), dim=0)
    
    # add edges
    graph.add_edges(src_tensor, dst_tensor, etype="ppi")

    assert new_weight.shape[0] == graph.num_edges('ppi'), f"Mismatch: {new_weight.shape[0]} vs {graph.num_edges('ppi')}"
    
    graph.edges['ppi'].data['w'] = new_weight
    return graph


def plot_sim_distribution(sim_dict, name):

    sim_values = np.array(list(sim_dict.values()))

    fig, ax_main = plt.subplots(figsize=(8, 6))

    bins_main = np.linspace(0, 1, 100)  
    ax_main.hist(sim_values, bins=bins_main, color='#826ba2', alpha=0.7, edgecolor='#826ba2', label="Full Range")
    ax_main.set_xlabel("Similarity", fontsize=12)
    ax_main.set_ylabel("Frequency", fontsize=12)

    ax_inset = inset_axes(ax_main, width="35%", height="35%", loc="center right", borderpad=3)

    bins_zoomed = np.linspace(0.9, 1, 50) 
    hist_inset = ax_inset.hist(sim_values, bins=bins_zoomed, color='#9a8ab4', alpha=0.7, edgecolor='#9a8ab4', label="Zoomed In")
    ax_inset.set_xlim(0.9, 1)
    ax_inset.set_xticks(np.linspace(0.9, 1, 5))
    max_y_inset = max(hist_inset[0])  
    ax_inset.set_yticks(np.linspace(0, max_y_inset, 4))
    ax_inset.set_title("Distribution in 0.9-1.0", fontsize=10, fontweight='bold', pad=5)

    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_inset.spines["top"].set_visible(True)
    ax_inset.spines["right"].set_visible(True)

    plt.savefig(f'jaccard_resnik_sim_figures/en_{name}.jpg')
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--datapath", type=str, default="data")
    parser.add_argument("-o","--ontology", type=str, default="mf")
    parser.add_argument("-n","--num_samples", type=int, default=2000)
    parser.add_argument("-t","--threshold", type=float, default=0.95)
    parser.add_argument("-a","--alpha", type=float, default=0.5)
    args = parser.parse_args()
    logger.info(args)

    datapath = args.datapath
    ont = args.ontology
    threshold = args.threshold
    alpha = args.alpha
    num_samples = args.num_samples

    uniprot2string = get_uniprot2string(datapath)
    pid2index = get_ppi_pid2index(datapath)
    label_dict = get_tax_pid_label_dict(f'{datapath}/{ont}/{ont}_train_go.txt')

    ic_values, golist = get_go_ic(f'{datapath}/{ont}/{ont}_go_ic.txt')
    start = time.perf_counter()
    pos_samples = {}
    # TAXIDS = ['7227','9606']
    spieces = TAXIDS
    if 'testdata' in datapath:
        spieces = TAXIDS_test
    for idx1 in range(len(spieces)):
        for idx2 in range(idx1+1, len(spieces)):
            pids1 = list(label_dict[spieces[idx1]].keys())
            pids2 = list(label_dict[spieces[idx2]].keys())
            logger.info(f'The number of {spieces[idx1]} proteins is {len(pids1)}')
            logger.info(f'The number of {spieces[idx2]} proteins is {len(pids2)}')
            jaccard_sim_dict = compute_cross_jaccard_similarity(label_dict[spieces[idx1]], label_dict[spieces[idx2]]) #label_matrix为csr格式
            resnik_sim_dict = compute_cross_resnik_similarity(label_dict[spieces[idx1]], label_dict[spieces[idx2]], ic_values)
            pairs_score = weighted_sum(jaccard_sim_dict, resnik_sim_dict, alpha)

            plot_sim_distribution(pairs_score, f'{spieces[idx1]}_{spieces[idx2]}_jaccard_resnik_{alpha}_{ont}')
            
            positive_pairs = {pair:sim for pair, sim in pairs_score.items() if sim >= threshold}
            if len(positive_pairs) > num_samples:
                sample_pairs = random.sample(list(positive_pairs.keys()), num_samples)
                positive_pairs = {pair:positive_pairs[pair] for pair in sample_pairs}

            for (p1,p2), score in positive_pairs.items():
                pos_samples[(pid2index[uniprot2string[p1]], pid2index[uniprot2string[p2]])] = score

    logger.info(f'Positive samples total {len(list(pos_samples.keys()))}')
    with open(f'tmp/Positive_samples_mf_a{alpha}_t{threshold}.txt','w') as f:
        for (i, j) in pos_samples:
            f.write(f'{i}\t{j}\n')

    logger.info(f'jaccard similarity done.')
    end = time.perf_counter()
    logger.info("Run time:{}".format(end-start)) 

    dgl_path = F'{datapath}/dgl_hetero'
    logger.info(F"Loading dgl graph: {dgl_path}......")
    g = dgl.load_graphs(dgl_path)[0][0]
    
    print(g)
    logger.info(F"Add new ppi to dgl graph......")
    g = add_ppi_edges(g, pos_samples)
    print(g)

    dgl.data.utils.save_graphs(F'{datapath}/dgl_hetero_{ont}_{alpha}', g)