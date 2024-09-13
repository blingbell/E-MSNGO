import argparse
from metrics import compute_metrics
from logzero import logger
import numpy as np
from dgl.dataloading import GraphDataLoader
import torch
from model import SAGNetworkHierarchical
import torch.nn as nn
import time
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import get_mlb, get_label_dict, get_go_ic, get_ppi_pid2index, get_uniprot2string
from torch.utils.data import Dataset

class MyDataSet(Dataset):
    def __init__(self,emb_graph,true_label):
        super().__init__()
        self.list = list(emb_graph.keys())
        self.graphs = emb_graph
        self.true_labels = true_label

    def __getitem__(self,index): 
        protein = self.list[index] 
        graph = self.graphs[protein]
        label = self.true_labels[protein]

        return protein, graph, label

    def __len__(self):
        return  len(self.list) 

def divide_dataset(ont:str, datapath:str, go_list):
    go_mlb = get_mlb(f'{datapath}/{ont}/{ont}_go.mlb', go_list)

    train_true_label = get_label_dict(f'{datapath}/{ont}/{ont}_train_go.txt')
    true_label_binary = go_mlb.transform(train_true_label.values()).astype(np.float32).toarray()
    assert len(true_label_binary) == len(train_true_label)
    for i, pid in enumerate(train_true_label.keys()):
        train_true_label[pid] = true_label_binary[i]

    valid_true_label = get_label_dict(f'{datapath}/{ont}/{ont}_valid_go.txt')
    valid_label_binary = go_mlb.transform(valid_true_label.values()).astype(np.float32).toarray()
    assert len(valid_label_binary) == len(valid_true_label)
    for i, pid in enumerate(valid_true_label.keys()):
        valid_true_label[pid] = valid_label_binary[i]
    
    test_true_label = get_label_dict(f'{datapath}/{ont}/{ont}_test_go.txt')
    test_label_binary = go_mlb.transform(test_true_label.values()).astype(np.float32).toarray()
    assert len(test_label_binary) == len(test_true_label)
    for i, pid in enumerate(test_true_label.keys()):
        test_true_label[pid] = test_label_binary[i]
    logger.info('load true labels done')

    train_pids = np.loadtxt(f'{datapath}/{ont}/{ont}_train_pids.txt', dtype=str) #uniprot id
    valid_pids = np.loadtxt(f'{datapath}/{ont}/{ont}_valid_pids.txt', dtype=str)
    test_pids = np.loadtxt(f'{datapath}/{ont}/{ont}_test_pids.txt', dtype=str)
    logger.info('load pids done')
    struct_emb = np.load(f'{datapath}/emb_graph_{ont}.npy', allow_pickle=True).item()
    logger.info('load struct emb done')

    train_struct_emb = {}
    valid_struct_emb = {}
    test_struct_emb = {}
    
    for pid in tqdm(struct_emb, desc = 'divide dataset...'):
        if pid in train_pids:
            train_struct_emb[pid] = struct_emb[pid]

        elif pid in valid_pids:
            valid_struct_emb[pid] = struct_emb[pid]

        elif pid in test_pids:

            test_struct_emb[pid] = struct_emb[pid]

    train_dataset = MyDataSet(train_struct_emb,  train_true_label)
    valid_dataset = MyDataSet(valid_struct_emb, valid_true_label)
    test_dataset = MyDataSet(test_struct_emb, test_true_label)
    logger.info(f'load {ont} datasets done')
    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    #参数设置
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datapath", type=str, default='data', help="path to save data")
    parser.add_argument("-m", "--modelpath", type=str, help="path to save model file")
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,  help="the number of the bach size")
    parser.add_argument('-lr', '--lr',type=float,default=5e-4)
    parser.add_argument('-dropout', '--dropout',type=float,default=0.45)
    parser.add_argument('-ont', '--ont',type=str,default='mf')
    parser.add_argument('-e', '--n_epochs',type=int,default=20)

    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    ont = args.ont
    datapath = args.datapath
    modelpath = args.modelpath
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    # device = torch.device('cuda:1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info(f'loading {ont} dataset...')

    go_ic, go_list = get_go_ic(f'{datapath}/{ont}/{ont}_go_ic.txt')
    go_mlb = get_mlb(f'{datapath}/{ont}/{ont}_go.mlb', go_list)
    label_classes = go_mlb.classes_
    labels_num = len(label_classes)

    train_dataset, valid_dataset, test_dataset = divide_dataset(ont, datapath, go_list)

    train_dataloader = GraphDataLoader(dataset=train_dataset, batch_size = batch_size)
    valid_dataloader = GraphDataLoader(dataset=valid_dataset, batch_size = batch_size)
    test_dataloader = GraphDataLoader(dataset=test_dataset, batch_size = batch_size)

    logger.info('#########'+ ont +'###########')
    logger.info('########start training struct model SAGNetworkHierarchical ###########')

    model = SAGNetworkHierarchical(56,512,labels_num,num_convs=2,pool_ratio=0.75,dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

    ce_loss_fn = nn.BCEWithLogitsLoss()
    # train
    best_fmax = 0.0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        for proteins, batched_graph, true_label in tqdm(train_dataloader, desc = f'train epoch {epoch} ...'):
            batch_tic = time.time()
            logits= model(batched_graph.to(device))
            loss = ce_loss_fn(logits, true_label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info("Epoch {:04d} | Train Loss: {:.4f} | Time: {:.4f}".format(epoch, loss.item(), time.time() - batch_tic))
        # eval
        model.eval()
        pred = []
        actual = []
        for proteins, batched_graph, true_label in tqdm(valid_dataloader, desc = f'eval epoch {epoch} ...'):
            batch_tic = time.time()
            # true_label = torch.reshape(true_label,(-1,labels_num))
            logits= model(batched_graph.to(device))
            loss = ce_loss_fn(logits, true_label.to(device))
            pred.extend(torch.sigmoid(logits).cpu().detach().numpy())
            actual.extend(true_label.detach().numpy())
        pred_matrix = np.array(pred)
        label_matrix = csr_matrix(np.array(actual))
        (fmax_, smin_, threshold), aupr_ = compute_metrics(label_matrix, pred_matrix, go_ic, label_classes)
        logger.info("Epoch {:04d} | Valid X Fmax: {:.4f} | Smin: {:.4f} | threshold: {:.2f} | AUPR: {:.4f}  Time: {:.4f} ".
                    format(epoch, fmax_, smin_, threshold, aupr_, time.time() - t0))
        if fmax_ > best_fmax:
            logger.info(F'improved from {best_fmax} to {fmax_}, save model to {modelpath}.')
            best_fmax = fmax_
            torch.save(model.state_dict(), '{}/{}_e{}_lr{}_dr{}_SAG.ckp'.format(modelpath, ont, args.n_epochs,lr,dropout))


    # encode
    logger.info('########start encode struct ###########')
    encoder = SAGNetworkHierarchical(56,512,labels_num,num_convs=2,pool_ratio=0.75,dropout=dropout).to(device)
    encoder.load_state_dict(torch.load('{}/{}_e{}_lr{}_dr{}_SAG.ckp'.format(modelpath, ont, args.n_epochs,lr,dropout)))
    ppi_pid2index = get_ppi_pid2index(datapath)
    uniprot2string = get_uniprot2string(datapath)
    struct_feat_matrix = np.zeros((len(ppi_pid2index),1024))
    for proteins, batched_graph, true_label in tqdm(train_dataloader, desc = f'encode trainset ...'):
        structemb = encoder.encode(batched_graph.to(device))
        structemb = structemb.cpu().detach().numpy()
        netids = [ppi_pid2index[uniprot2string[protein]] for protein in proteins]
        struct_feat_matrix[netids] = structemb

    for proteins, batched_graph, true_label in tqdm(valid_dataloader, desc = f'encode validset ...'):
        structemb = encoder.encode(batched_graph.to(device))
        structemb = structemb.cpu().detach().numpy()
        netids = [ppi_pid2index[uniprot2string[protein]] for protein in proteins]
        struct_feat_matrix[netids] = structemb

    for proteins, batched_graph, true_label in tqdm(test_dataloader, desc = f'encode testset ...'):
        structemb = encoder.encode(batched_graph.to(device))
        structemb = structemb.cpu().detach().numpy()
        netids = [ppi_pid2index[uniprot2string[protein]] for protein in proteins]
        struct_feat_matrix[netids] = structemb
    
    np.save(f'{datapath}/sag_struct_feature_{ont}.npy', struct_feat_matrix)
    logger.info(f'save {ont} struct sag model emb')