import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools

from data import DataProvider
import train_coral
import train_dann
import train_dcc
import train_adda
from copy import deepcopy


def generate_encoded_features(encoder, dataloader, normalize_flag=False):
    """

    :param normalize_flag:
    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0].cpu()
    label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor, label_tensor


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])


def main(args):
    if args.method == 'dann':
        train_fn = train_dann.train_dann
    elif args.method == 'adda':
        train_fn = train_adda.train_adda
    elif args.method == 'dcc':
        train_fn = train_dcc.train_dcc
    else:
        train_fn = train_coral.train_coral

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join('model_save', 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', args.method, 'labeled'),
        })

    safe_make_dir(training_params['model_save_folder'])
    data_provider = DataProvider(batch_size=training_params['labeled']['batch_size'],
                                 target=args.measurement)
    training_params.update(
        {
            'input_dim': data_provider.shape_dict['gex'],
            'output_dim': data_provider.shape_dict['target']
        }
    )

    s_labeled_dataloader = data_provider.get_labeled_gex_dataloader()
    labeled_dataloader_generator = data_provider.get_drug_labeled_mut_dataloader()

    s_ft_evaluation_metrics = defaultdict(list)
    t_ft_evaluation_metrics = defaultdict(list)
    val_ft_evaluation_metrics = defaultdict(list)
    test_ft_evaluation_metrics = defaultdict(list)

    fold_count = 0
    for train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader in labeled_dataloader_generator:
        target_regressor, train_historys = train_fn(s_dataloaders=s_labeled_dataloader,
                                                    t_dataloaders=train_labeled_dataloader,
                                                    val_dataloader=val_labeled_dataloader,
                                                    test_dataloader=test_labeled_dataloader,
                                                    metric_name=args.metric,
                                                    seed = fold_count,
                                                    **wrap_training_params(training_params, type='labeled'))

        for metric in ['dpearsonr', 'dspearmanr', 'drmse', 'cpearsonr', 'cspearmanr', 'crmse']:
            val_ft_evaluation_metrics[metric].append(train_historys[-2][metric][train_historys[-2]['best_index']])
            test_ft_evaluation_metrics[metric].append(train_historys[-1][metric][train_historys[-2]['best_index']])
        fold_count += 1

    with open(os.path.join(training_params['model_save_folder'], f'test_ft_evaluation_results.json'), 'w') as f:
        json.dump(test_ft_evaluation_metrics, f)
    with open(os.path.join(training_params['model_save_folder'], f'ft_evaluation_results.json'), 'w') as f:
        json.dump(val_ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Domain adaptation training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='coral',
                        choices=['dcc', 'dann', 'coral', 'adda'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='cpearsonr', choices=['cpearsonr', 'dpearsonr'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    args = parser.parse_args()

    main(args=args)
