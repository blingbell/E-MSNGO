import networkx as nx
import matplotlib.pyplot as plt
import torch
from GNNExplainer import GNNExplainer
import argparse
from logzero import logger
import numpy as np
import torch
import dgl
import dgl.data.utils
import os
from utils import *
import yaml
from model import MSNGO

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def feature_hotmap(vector, file):
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    # vector = np.loadtxt("/e/wangbb/MSNGO/case_result/explain_result/19319_sim_feature.txt")  

    sequence_vector = vector[:1280]  # seq
    structure_vector = vector[1280:]  # struct

    sequence_vector_2d = sequence_vector.reshape(32, 40)  # 1280 = 32x40
    structure_vector_2d = structure_vector.reshape(32, 32)  # 1024 = 32x32

    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

    im1 = axes[0].imshow(sequence_vector_2d, aspect='auto', cmap='plasma')
    axes[0].set_title("Heat map of sequence features", fontsize = 17)
    fig.colorbar(im1, ax=axes[0])  

    im2 = axes[1].imshow(structure_vector_2d, aspect='auto', cmap='plasma')
    axes[1].set_title("Heat map of structure features", fontsize = 17)
    fig.colorbar(im2, ax=axes[1])  

    plt.tight_layout()
    plt.savefig(f'{file}.jpg')


def explain_msngo(model, dataset, device, ont, label, result_dir, num_nodes):
    
    g = dataset['g']
    seq_feature_matrix = dataset['seq_feature']
    struct_feature_matrix = dataset['struct_feature']
    label_matrix = dataset['label']
    # select nodes randomly or u can choose the node id in dataset['idx'] like next line
    selected_nodes = [int(node) for node in np.random.choice(dataset['idx'], num_nodes, replace=False)]
    # selected_nodes = [11360, 19319, 83771 ,51762, 18725, 50971]
    print(f'selected_nodes = {selected_nodes}')
        
    nodes_index = np.asarray(selected_nodes).astype(np.int32)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) 
    dataloader = dgl.dataloading.DataLoader(g, nodes_index, sampler, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    
    explainer = GNNExplainer(model, epochs=100, log=False)

    for input_nodes, _, blocks in dataloader:
        blocks = [blk.to(device) for blk in blocks] 
        batch_seq = torch.from_numpy(seq_feature_matrix[input_nodes.numpy()]).float().to(device)
        batch_struct = torch.from_numpy(struct_feature_matrix[input_nodes.numpy()]).float().to(device)
        batch_x = torch.cat((batch_seq, batch_struct), dim=-1)
        batch_y = None
        if label:
            batch_y = torch.from_numpy(label_matrix[input_nodes.numpy()]).float().to(device)

        feature_mask_ppi, edge_mask_ppi = explainer.explain_node(input_nodes[0].item(), blocks, batch_x, batch_y, etype="ppi")
        important_ppi_edges = extract_important_edges(result_dir, blocks, edge_mask_ppi, "ppi", ont)

        feature_mask_sim, edge_mask_sim = explainer.explain_node(input_nodes[0].item(), blocks, batch_x, batch_y, etype="sim")
        important_sim_edges = extract_important_edges(result_dir, blocks, edge_mask_sim, "sim", ont)

        feature_hotmap(feature_mask_ppi.cpu().numpy().tolist(), f'{result_dir}/{ont}_{input_nodes[0].item()}_PPI')
        # np.savetxt(f'case_result/explain_result/{input_nodes[0].item()}_PPI_feature.txt', feature_mask_ppi.cpu().numpy())

        feature_hotmap(feature_mask_sim.cpu().numpy().tolist(), f'{result_dir}/{ont}_{input_nodes[0].item()}_SIM')
        # np.savetxt(f'case_result/explain_result/{input_nodes[0].item()}_sim_feature.txt', feature_mask_sim.cpu().numpy())

        print(f"key ppi links: {important_ppi_edges}")
        print(f"key sim links: {important_sim_edges}")
        visualize_edges(important_sim_edges, title=f'{result_dir}/{ont}_sim_edges_{input_nodes[0].item()}')
        visualize_edges(important_ppi_edges, title=f'{result_dir}/{ont}_ppi_edges_{input_nodes[0].item()}')

def extract_important_edges(result_dir, blocks, edge_mask, etype, ont, threshold=0.5):
    important_edges = []
    edge_size = [0]
    
    for i in range(len(blocks)):
        subgraph = blocks[i]
        ntype = subgraph.ntypes[0]
        nodeids = subgraph.srcnodes[ntype].data[dgl.NID]
        edge_size.append(subgraph.edges(etype=etype)[0].size(0))
        edge_index = subgraph.edges(etype=etype)  
        with open(f'{result_dir}/{ont}_edge_weight_{etype}_{nodeids[0].item()}.txt','w') as f:
            for j, weight in enumerate(edge_mask[edge_size[i]:edge_size[i]+edge_size[i+1]]):
                src, dst = edge_index[0][j].item(), edge_index[1][j].item()
                f.write(f'{nodeids[src].item()}\t{nodeids[dst].item()}\t{weight.item()}\n')
                if weight > threshold:
                    important_edges.append((nodeids[src].item(),nodeids[dst].item(), weight.item()))
                
    
    return important_edges


def visualize_edges(edges, title):
    if not edges:
        print(" no importance edges")
        return

    G = nx.Graph()
    for src, dst, weight in edges:
        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))

    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color=weights,
            edge_cmap=plt.cm.Blues, width=2, font_size=10)
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(e[0], e[1]): f"{w:.2f}" for e, w in zip(edges, weights)},
                                 font_size=8, font_color='red')

    plt.title(title)
    plt.savefig(f'{title}.jpg')

def main(args):
    data_dir = args.datapath
    ont = args.ontology
    alpha = args.alpha
    result_dir = args.result_dir
    num_nodes = args.num_nodes
    uniprot2string = get_uniprot2string(data_dir)
    pid2index = get_ppi_pid2index(data_dir)

    dgl_path = F'{data_dir}/dgl_hetero_{ont}_{alpha}'
    g = dgl.load_graphs(dgl_path)[0][0]

    train_pid_list, train_label_list = get_pid_and_label_list(f'{data_dir}/{ont}/{ont}_train_go.txt')

    train_idx, train_pid_list, train_label_list = get_network_index(uniprot2string, pid2index, train_pid_list, train_label_list)

    seq_feature_matrix = np.load(f'{data_dir}/esm_feature.npy')
    struct_feature_matrix = np.load(f'{data_dir}/sag_struct_feature_{ont}.npy')
    
    go_ic, go_list = get_go_ic(os.path.join(data_dir, ont, f'{ont}_go_ic.txt'))
    go_mlb = get_mlb(os.path.join(data_dir, ont, f'{ont}_go.mlb'), go_list)
    label_classes = go_mlb.classes_
    train_y = go_mlb.transform(train_label_list).astype(np.float32)
    label_matrix = np.zeros((seq_feature_matrix.shape[0], label_classes.shape[0]))
    label_matrix[train_idx] = train_y.toarray()
    flag = np.full(seq_feature_matrix.shape[0], False, dtype=bool)
    flag[train_idx] = np.full(train_idx.shape[0], True, dtype=bool)
    g.ndata['flag'] = torch.from_numpy(flag)


    # load model config 
    with open('./MSNGO.yaml', 'r', encoding='UTF-8') as f:
        model_config = yaml.load(f, yaml.FullLoader)
    make_diamond_db(os.path.join(data_dir, 'network.fasta'), os.path.join(data_dir, 'network_db.dmnd'))
    if args.input_file == None:
        pred_pid_list = get_pid_list(os.path.join(data_dir, ont, f'{ont}_test.fasta'))
        pred_diamond_result  = diamond_homo(os.path.join(data_dir, 'network_db.dmnd'),
                                            os.path.join(data_dir, ont, f'{ont}_test.fasta'),
                                            os.path.join(data_dir, ont, f'{ont}_test_diamond.txt'))
    else:
        pred_pid_list = get_pid_list(args.input_file)
        filename = args.input_file.split('.')[0]
        pred_diamond_result  = diamond_homo(os.path.join(data_dir, 'network_db.dmnd'), args.input_file,
                                            os.path.join(result_dir, filename+'_diamond.txt'))
    pred_index = list()
    tmp_pid_list = list()
    ct = 0
    for pid in pred_pid_list:
        if pid2index.get(pid) != None:
            pred_index.append(pid2index[pid])
            tmp_pid_list.append(pid)
        elif uniprot2string.get(pid) != None and pid2index.get(uniprot2string[pid]) != None:
            pred_index.append(pid2index[uniprot2string[pid]])
            tmp_pid_list.append(pid)
        elif pred_diamond_result.get(pid) != None:
            pred_index.append( pid2index[ max( pred_diamond_result[pid].items(), key=lambda x: x[1] )[0] ] )
            tmp_pid_list.append(pid)
            ct += 1
    logger.info(F"There are {ct} proteins that don't have network index.")

    #device = 'cuda:1'
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:%d' % args.gpu

    logger.info('Load trained model.')
    model = MSNGO(seq_feature_matrix.shape[1]+struct_feature_matrix.shape[1], model_config['n_hidden'], label_matrix.shape[1],
            model_config['n_mlp_layers'], model_config['n_prop_steps'], mlp_drop=model_config['mlp_dropout'],
            residual=model_config['residual'], share_weight=False).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, F'E-MSNGO_{ont}_{args.model_id}.ckp'))) 
    
    testid = np.unique(pred_index)
    testid = testid.astype(np.int32)
    dataset = dict()
    dataset['idx'] = testid
    dataset['g'], dataset['seq_feature'], dataset['struct_feature'], dataset['label'],  = g, seq_feature_matrix, struct_feature_matrix, label_matrix
    explain_msngo(model, dataset, device,ont, args.label, result_dir, num_nodes=num_nodes)
    logger.info('Finished Explain.\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explainer.')
    parser.add_argument("-d","--datapath", type=str, default="data")
    parser.add_argument("-o","--ontology", type=str, default="mf")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("-a","--alpha", type=float, default=0.5,
                        help="choose dgl graph")
    parser.add_argument("-n","--num_nodes", type=float, default=50,
                        help="The number of nodes (randomly choosed) to explain")
    parser.add_argument("--label", type=bool, default=False,
                        help="label propagation: False for test ids, True for train and valid ids. ")
    parser.add_argument("--model_dir", type=str, default='./MSNGO_models',
                        help='path for save the model parameters')
    parser.add_argument("--model_id", type=str, default='0')
    parser.add_argument("-r", "--result_dir", type=str, default='explain_result',
                        help="The file path of the prediction results.")
    parser.add_argument("-f", "--input_file", type=str, default=None,
                        help="The fasta file path of the protein that needs to be predicted.")

    
    args = parser.parse_args()
    logger.info("Running the MSNGO model for prediction.")
    logger.info(F"Ontology: {args.ontology}")
    logger.info(F"Model ID: {args.model_id}")
    logger.info(F"alpha: {args.alpha}")
    logger.info(F"GPU: {args.gpu}")
    logger.info(F"Result dir: {args.result_dir}")
    logger.info(F"The input FASTA file path: {args.input_file}")
    main(args)