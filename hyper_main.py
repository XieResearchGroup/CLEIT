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
import data_config
import train_coral
import fine_tuning
import train_dsn
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


def main(args, update_params_dict):
    if args.method == 'cleitm':
        raise NotImplementedError
    elif args.method == 'cleita':
        raise NotImplementedError
    elif args.method == 'coral':
        train_fn = train_coral.train_coral
    elif args.method == 'dsn':
        train_fn = train_dsn.train_dsn
    elif args.method == 'dann':
        raise NotImplementedError
    elif args.method == 'adda':
        raise NotImplementedError
    elif args.method == 'dcc':
        raise NotImplementedError
    else:
        raise NotImplementedError

    # normalize_flag = args.method in ['adsn', 'mdsn', 'ndsn']
    # normalize_flag = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(os.path.join('model_save', 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', args.method, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag
        })
    task_save_folder = os.path.join('model_save', args.method, args.measurement, args.drug)

    safe_make_dir(training_params['model_save_folder'])
    safe_make_dir(task_save_folder)

    data_provider = DataProvider(batch_size=training_params['unlabeled']['batch_size'],
                                 target=args.measurement)
    training_params.update(
        {
            'input_dim': data_provider.shape_dict['gex'],
        }
    )

    random.seed(2020)

    # start unlabeled training
    encoder, historys = train_fn(s_dataloaders=data_provider.get_unlabeld_mut_dataloader(match=True),
                                 t_dataloaders=data_provider.get_unlabeled_gex_dataloader(),
                                 **wrap_training_params(training_params, type='unlabeled'))
    with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)
    labeled_dataloader_generator = data_provider.get_drug_labeled_mut_dataloader(drug=args.drug)
    ft_evaluation_metrics = defaultdict(list)
    test_ft_evaluation_metrics = defaultdict(list)
    fold_count = 0
    for train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader in labeled_dataloader_generator:
        ft_encoder = deepcopy(encoder)
        target_classifier, ft_historys = fine_tuning.fine_tune_encoder(
                encoder=ft_encoder,
                train_dataloader=train_labeled_dataloader,
                val_dataloader=val_labeled_dataloader,
                test_dataloader=test_labeled_dataloader,
                seed=fold_count,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                **wrap_training_params(training_params, type='labeled')
        )
        for metric in ['pearsonr', 'spearmanr', 'r2', 'rmse']:
            ft_evaluation_metrics[metric].append(ft_historys[-2][metric][ft_historys[-2]['best_index']])
            test_ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
        fold_count += 1
    with open(os.path.join(task_save_folder, f'{param_str}_test_ft_evaluation_results.json'), 'w') as f:
        json.dump(test_ft_evaluation_metrics, f)
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLEIT training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='coral',
                        choices=['cleit', 'cleita', 'cleitm', 'dsn', 'dcc', 'dann', 'coral', 'adaa'])
    parser.add_argument('--drug', dest='drug', nargs='?', default='gem', choices=['gem', 'fu', 'cis', 'tem'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='pearsonr', choices=['pearsonr', 'rmse'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    args = parser.parse_args()

    params_grid = {
        #"pretrain_num_epochs": [0, 50, 100, 200, 300],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
        "dop": [0.0, 0.1, 0.2]
        #"train_num_epochs": [100]
    }

    # if args.method not in ['adsn', 'adae', 'dsnw']:
    #     params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict)
