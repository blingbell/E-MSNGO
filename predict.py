

import argparse
from logzero import logger
import numpy as np
import torch
import dgl
import dgl.data.utils
import os
from utils import *
from model import MSNGO
import yaml


def main(args):
    data_dir = args.datapath
    result_dir = args.result_dir
    ont = args.ontology
    alpha = args.alpha
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

    pred_x_score_list, pred_y_score_list = list(), list()
    logger.info('Start predict......')
    logger.info('Load trained model.')
    model = MSNGO(seq_feature_matrix.shape[1]+struct_feature_matrix.shape[1], model_config['n_hidden'], label_matrix.shape[1],
            model_config['n_mlp_layers'], model_config['n_prop_steps'], mlp_drop=model_config['mlp_dropout'],
            residual=model_config['residual'], share_weight=model_config['share_weight']).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, F'MSNGO_{args.ontology}_{args.model_id}.ckp'))) 
    pred_x_score_list, pred_y_score_list = model.inference(g, pred_index, seq_feature_matrix, struct_feature_matrix, label_matrix, model_config['batch_size'], device)

    alpha = model_config['alphas'][ont]
    pred_combine_score = alpha * pred_x_score_list + (1 - alpha) * pred_y_score_list

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if args.input_file == None:
        with open(os.path.join(result_dir, F'{ont}_prediction_MSNGO_{args.model_id}.txt'), 'w') as f:
            for i in range(pred_combine_score.shape[0]):
                for j in range(pred_combine_score.shape[1]):
                    f.write(F'{tmp_pid_list[i]}\t{label_classes[j]}\t{pred_combine_score[i, j]}\n')
    else:
        with open(os.path.join(result_dir, args.input_file+f'_MSNGO_prediction.txt'), 'w') as f:
            for i in range(pred_combine_score.shape[0]):
                for j in range(pred_combine_score.shape[1]):
                    f.write(F'{tmp_pid_list[i]}\t{label_classes[j]}\t{pred_combine_score[i, j]}\n')
    logger.info(F'Finished save predicted result.\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction.')
    parser.add_argument("-d","--datapath", type=str, default="data")
    parser.add_argument("--ontology", type=str, default="bp")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--model_dir", type=str, default='./MSNGO_models',
                        help='path for save the model parameters')
    parser.add_argument("-a","--alpha", type=float, default=0.5,
                        help="choose dgl graph")
    parser.add_argument("-f", "--input_file", type=str, default=None,
                        help="The fasta file path of the protein that needs to be predicted.")
    parser.add_argument("-r", "--result_dir", type=str, default='result',
                        help="The file path of the prediction results.")
    parser.add_argument("--model_id", type=str, default='0')
    args = parser.parse_args()
    logger.info("Running the MSNGO model for prediction.")
    logger.info(F"Ontology: {args.ontology}")
    logger.info(F"GPU: {args.gpu}")
    logger.info(F"Results in : {args.result_dir}")
    logger.info(F"The input FASTA file path: {args.input_file}")
    main(args)

######################################################################### end ################################################################################
