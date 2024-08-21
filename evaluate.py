

import argparse
from utils import get_mlb, get_pred_matrix, get_pid_and_label_list, get_go_ic
from logzero import logger
import os
from metrics import compute_metrics




def main(args):
    data_dir = args.datapath
    result_dir = args.result_dir
    ont = args.ontology

    logger.info(F'Evaluate MSNGO model.')
    go_ic, go_list = get_go_ic(os.path.join(data_dir, ont, f'{ont}_go_ic.txt'))
    mlb = get_mlb(os.path.join(data_dir, ont, f'{ont}_go.mlb'), go_list)
    pid_list, label_list = get_pid_and_label_list(f'{data_dir}/{ont}/{ont}_test_go.txt')
    print('len(pid_list)',len(pid_list))
    pid2index = {pid: i for i, pid in enumerate(pid_list)}
    label_matrix = mlb.transform(label_list)
    label_classes = mlb.classes_
    go2index = {go_id: i for i, go_id in enumerate(label_classes)}
    print('len(go2index)',len(go2index))
    pred_matrix = get_pred_matrix(os.path.join(result_dir, F'{ont}_prediction_MSNGO_{args.model_id}.txt'), pid2index, go2index)
    print(pred_matrix.shape)
    print(label_matrix.todense().shape)
    (fmax_, smin_, threshold), aupr_ = compute_metrics(label_matrix, pred_matrix, go_ic, label_classes)
    logger.info("Aspect: {} | Fmax: {:.4f} | Smin: {:.4f} | AUPR: {:.4f} | threshold: {:.2f}\n".format(ont, fmax_, smin_, aupr_, threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("-d", "--datapath", type=str, default='data')
    parser.add_argument("--ontology", type=str, default='mf')
    parser.add_argument("-r", "--result_dir", type=str, default='result',
                        help="The file path of the prediction results.")
    parser.add_argument("--model_id", type=str, default='0')

    args = parser.parse_args()
    logger.info(args)
    main(args)