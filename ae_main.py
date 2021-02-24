import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools

from data import DataProvider
import train_ae
import fine_tuning

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
    train_fn = train_ae.train_ae
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join('model_save', 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', 'ae', args.omics, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag
        })
    safe_make_dir(training_params['model_save_folder'])
    task_save_folder = os.path.join('model_save', 'ae', args.omics, param_str)
    safe_make_dir(task_save_folder)

    random.seed(2020)

    data_provider = DataProvider(batch_size=training_params['unlabeled']['batch_size'],
                                 target=args.measurement)

    training_params.update(
        {
            'input_dim': data_provider.shape_dict[args.omics],
            'output_dim': data_provider.shape_dict['target']
        }
    )
    # start unlabeled training
    if args.omics == 'gex':
        encoder, historys = train_fn(dataloader=data_provider.get_unlabeled_gex_dataloader(),
                                     **wrap_training_params(training_params, type='unlabeled'))
    else:
        encoder, historys = train_fn(dataloader=data_provider.get_unlabeld_mut_dataloader(match=False),
                                     **wrap_training_params(training_params, type='unlabeled'))

    with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)

    ft_evaluation_metrics = defaultdict(list)
    fold_count = 0
    if args.omics == 'gex':
        labeled_dataloader_generator = data_provider.get_labeled_data_generator(omics='gex')
        for train_labeled_dataloader, val_labeled_dataloader in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            target_regressor, ft_historys = fine_tuning.fine_tune_encoder(
                encoder=ft_encoder,
                train_dataloader=train_labeled_dataloader,
                val_dataloader=val_labeled_dataloader,
                test_dataloader=val_labeled_dataloader,
                seed=fold_count,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'dspearmanr','drmse', 'cpearsonr', 'cspearmanr','crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-2][metric][ft_historys[-2]['best_index']])
            fold_count += 1
    else:
        labeled_dataloader_generator = data_provider.get_labeled_data_generator(omics='mut')
        test_ft_evaluation_metrics = defaultdict(list)
        for train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            target_regressor, ft_historys = fine_tuning.fine_tune_encoder(
                encoder=ft_encoder,
                train_dataloader=train_labeled_dataloader,
                val_dataloader=val_labeled_dataloader,
                test_dataloader=test_labeled_dataloader,
                seed=fold_count,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'dspearmanr','drmse', 'cpearsonr', 'cspearmanr','crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-2][metric][ft_historys[-2]['best_index']])
                test_ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
            fold_count += 1
        with open(os.path.join(task_save_folder, f'{param_str}_test_ft_evaluation_results.json'), 'w') as f:
            json.dump(test_ft_evaluation_metrics, f)
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLEIT training and evaluation')
    parser.add_argument('--omics', dest='omics', nargs='?', default='gex',
                        choices=['gex', 'mut'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='cpearsonr', choices=['cpearsonr', 'dpearsonr'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    args = parser.parse_args()

    params_grid = {
        #"pretrain_num_epochs": [0, 50, 100, 200, 300],
        #"train_num_epochs": [100, 300, 500, 1000, 2000, 3000, 5000],
        "train_num_epochs": [100, 300, 500, 1000, 2000, 3000, 5000],
        "dop": [0.0, 0.1]
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict)
