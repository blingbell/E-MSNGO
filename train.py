""" train MSNGO model. """


import argparse
from logzero import logger
import scipy.sparse as ssp
import numpy as np
import time
import torch
import torch.nn as nn
import dgl
import dgl.data.utils
import os
from utils import *
from metrics import compute_metrics
from model import MSNGO
import yaml


def train(args, dataset):
    device = torch.device('cuda:' + str(args.gpu))
    # dataset
    g, seq_feature_matrix, struct_feature_matrix, label_matrix, train_idx = dataset['g'], dataset['seq_feature'], dataset['struct_feature'], dataset['label'], dataset['train_idx'].astype(np.int32)
    valid_idx, valid_y = dataset['valid_idx'].astype(np.int32), dataset['valid_y'] 
    go_ic, label_classes = dataset['goic'], dataset['label_classes']

    model = MSNGO(seq_feature_matrix.shape[1]+struct_feature_matrix.shape[1], args.n_hidden, label_matrix.shape[1],
                args.n_mlp_layers, args.n_prop_steps, mlp_drop=args.mlp_dropout,
                residual=args.residual, share_weight=args.share_weight).to(device)
    # loss function
    ce_loss_fn = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # training dataloader
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_prop_steps)
    train_dataloader = dgl.dataloading.DataLoader(g, train_idx, sampler, batch_size=args.batch_size, 
                                                      shuffle=True, num_workers=2,  drop_last=False)
    # train
    best_fmax = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        with train_dataloader.enable_cpu_affinity():
            for i, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_tic = time.time()
                blocks = [blk.to(device) for blk in blocks]
                batch_seq = torch.from_numpy(seq_feature_matrix[input_nodes.numpy()]).float().to(device)
                batch_struct = torch.from_numpy(struct_feature_matrix[input_nodes.numpy()]).float().to(device)
                batch_x = torch.cat((batch_seq, batch_struct), dim=-1)
                batch_y = torch.from_numpy(label_matrix[seeds.numpy()]).float().to(device)
                logits, _ = model(blocks, batch_x)
                loss = ce_loss_fn(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    logger.info("Epoch {:04d} | Batch {:04d} | Train Loss: {:.4f} | Time: {:.4f}".
                            format(epoch, i, loss.item(), time.time() - batch_tic))
        # eval
        model.eval()
        unique_idx = np.unique(valid_idx)
        index_mapping = {idx: i for i, idx in enumerate(unique_idx)}
        res_idx = np.asarray([ index_mapping[idx] for idx in valid_idx ])
        valid_dataloader = dgl.dataloading.DataLoader(g, unique_idx, sampler,
                                     batch_size=args.batch_size, shuffle=False, num_workers=0,  drop_last=False)
        pred_list = []
        for input_nodes, _, blocks in valid_dataloader:
            blocks = [blk.to(device) for blk in blocks]
            batch_seq = torch.from_numpy(seq_feature_matrix[input_nodes.numpy()]).float().to(device)
            batch_struct = torch.from_numpy(struct_feature_matrix[input_nodes.numpy()]).float().to(device)
            batch_x = torch.cat((batch_seq, batch_struct), dim=-1)
            batch_pred, _ = model(blocks, batch_x)
            pred_list.append(torch.sigmoid(batch_pred).cpu().detach().numpy())
        valid_pred = np.vstack(pred_list)[res_idx]
        (fmax_, smin_, threshold), aupr_ = compute_metrics(valid_y, valid_pred, go_ic, label_classes)
        logger.info("Epoch {:04d} | Valid X Fmax: {:.4f} | Smin: {:.4f} | threshold: {:.2f} | AUPR: {:.4f}  Time: {:.4f} ".
                    format(epoch, fmax_, smin_, threshold, aupr_, time.time() - t0))
        if fmax_ > best_fmax:
            logger.info(F'improved from {best_fmax} to {fmax_}, save model to {args.model_dir} {args.model_id}.')
            best_fmax = fmax_
            torch.save(model.state_dict(), os.path.join(args.model_dir, F"MSNGO_{args.ontology}_{args.model_id}.ckp"))


def main(args):
    data_dir = args.datapath
    ont = args.ontology

    logger.info("Start loading data.")

    uniprot2string = get_uniprot2string(data_dir)
    pid2index = get_ppi_pid2index(data_dir)

    dgl_path = F'{data_dir}/dgl_hetero'
    logger.info(F"Loading dgl graph: {dgl_path}......")
    g = dgl.load_graphs(dgl_path)[0][0]
    logger.info(F"The info of the heterogeneous network: {g}")

    logger.info("Loading annotation data......")
    train_pid_list, train_label_list = get_pid_and_label_list(f'{data_dir}/{ont}/{ont}_train_go.txt')
    valid_pid_list, valid_label_list = get_pid_and_label_list(f'{data_dir}/{ont}/{ont}_valid_go.txt')
    logger.info(F"Number of train pid: {len(train_pid_list)}, valid pid: {len(valid_pid_list)}.")

    logger.info("Get Diamond result of valid set.")
    make_diamond_db(os.path.join(data_dir, 'network.fasta'), os.path.join(data_dir, 'network_db.dmnd'))
    valid_diamond_result = diamond_homo(os.path.join(data_dir, 'network_db.dmnd'),
                                        os.path.join(data_dir, ont, f'{ont}_valid.fasta'),
                                        os.path.join(data_dir, ont, f'{ont}_valid_diamond.txt'))

    logger.info("Mapping pid to network index.")
    train_idx, train_pid_list, train_label_list = get_network_index(uniprot2string, pid2index, train_pid_list, train_label_list)
    valid_idx, valid_pid_list, valid_label_list = get_network_index(uniprot2string, pid2index, valid_pid_list, valid_label_list,
                                                                    valid_diamond_result, category='eval')
    logger.info(F"Number of train index: {len(train_idx)}, valid index: {len(valid_idx)}.")
    
    logger.info('Get feature matrix.')
    struct_feature_matrix = np.load(f'/home/wangbb/mycode/dataset/sag_struct_feature_{ont}.npy')
    seq_feature_matrix = np.load(f'/home/wangbb/mycode/dataset/esm_feature.npy')
    logger.info(F'Shape of seq feature matrix: {seq_feature_matrix.shape}')
    logger.info(F'Shape of struct feature matrix: {struct_feature_matrix.shape}')

    logger.info('Get label matrix.')
    go_ic, go_list = get_go_ic(os.path.join(data_dir, ont, f'{ont}_go_ic.txt'))
    go_mlb = get_mlb(os.path.join(data_dir, ont, f'{ont}_go.mlb'), go_list)
    label_classes = go_mlb.classes_
    train_y = go_mlb.transform(train_label_list).astype(np.float32)
    valid_y = go_mlb.transform(valid_label_list).astype(np.float32)
    label_matrix = np.zeros((seq_feature_matrix.shape[0], label_classes.shape[0]))
    label_matrix[train_idx] = train_y.toarray()
    flag = np.full(seq_feature_matrix.shape[0], False, dtype=bool)
    flag[train_idx] = np.full(train_idx.shape[0], True, dtype=bool)
    g.ndata['flag'] = torch.from_numpy(flag)

    dataset = dict()
    dataset['g'], dataset['seq_feature'], dataset['struct_feature'], dataset['label'], dataset['train_idx'] = g, seq_feature_matrix, struct_feature_matrix, label_matrix, train_idx
    dataset['valid_idx'], dataset['valid_y'] = valid_idx, valid_y
    dataset['goic'], dataset['label_classes'] = go_ic, label_classes

    logger.info("Data loading is complete.")
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    logger.info("Start training...")
    train(args, dataset)
    logger.info('Finished train.\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train MSNGO model.')
    parser.add_argument("-d","--datapath", type=str, default="data")
    parser.add_argument("--ontology", type=str, default="mf")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--n_mlp_layers", type=int, default=1)
    parser.add_argument("--n_prop_steps", type=int, default=2)
    parser.add_argument("-e", "--n_epochs", type=int, default=10,
                        help="number of training epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=40,
                        help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--mlp_dropout", type=float, default=0.5,
                        help="mlp dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n_hidden", type=int, default=512,
                        help="hidden size")
    parser.add_argument("--residual", type=bool, default=True,
                        help="whether to make residual connection")
    parser.add_argument("--share_weight", type=bool, default=True,
                        help="whether parameters are shared between different types of networks")
    
    parser.add_argument("--model_id", type=str, default='0')
    parser.add_argument("--model_dir", type=str, default='./MSNGO_models',
                        help='path for save the model parameters')
    
    args = parser.parse_args()

    logger.info("Running the training script for MSNGO model.")
    logger.info(F"Ontology: {args.ontology}")
    logger.info(F"Data path: {args.datapath}")
    logger.info(F"Hyperparameters: ")
    logger.info(F"\t* Training epoch: {args.n_epochs}")
    logger.info(F"\t* Batch size: {args.batch_size}")
    logger.info(F"\t* Learning rate: {args.lr}")
    logger.info(F"\t* MLP layers: {args.n_mlp_layers}")
    logger.info(F"\t* Propagation layers: {args.n_prop_steps}")
    logger.info(F"\t* MLP dropout: {args.mlp_dropout}")
    main(args)


############################################################################### end ################################################################################