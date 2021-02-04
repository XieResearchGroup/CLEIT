import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict

from data import DataProvider
import train_vae
import fine_tuning
import parsing_utils

from copy import deepcopy


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


def build_encoder(args):
    train_fn = train_vae.train_vae
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join('model_save', 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    parsed_ft_params = parsing_utils.parse_hyper_vae_ft_evaluation_result(metric_name=args.metric)
    training_params['unlabeled'].update(parsed_ft_params[0])
    training_params['labeled']['train_num_epochs'] = parsed_ft_params[1]

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', 'vae'),
            'es_flag': False,
            'retrain_flag': True
        })

    task_save_folder = training_params['model_save_folder']
    safe_make_dir(training_params['model_save_folder'])

    random.seed(2020)

    data_provider = DataProvider(batch_size=training_params['unlabeled']['batch_size'],
                                 target=args.measurement)
    training_params.update(
        {
            'input_dim': data_provider.shape_dict['gex'],
            'output_dim': data_provider.shape_dict['target']
        }
    )
    encoder, historys = train_fn(dataloader=data_provider.get_unlabeled_gex_dataloader(),
                                 **wrap_training_params(training_params, type='unlabeled'))

    with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)

    ft_evaluation_metrics = defaultdict(list)
    labeled_dataloader = data_provider.get_labeled_gex_dataloader()
    ft_encoder = deepcopy(encoder)
    target_classifier, ft_historys = fine_tuning.fine_tune_encoder_new(
            encoder=ft_encoder,
            train_dataloader=labeled_dataloader,
            val_dataloader=labeled_dataloader,
            test_dataloader=None,
            seed=2021,
            metric_name=args.metric,
            task_save_folder=task_save_folder,
            **wrap_training_params(training_params, type='labeled')
        )
    for metric in ['dpearsonr', 'drmse', 'cpearsonr', 'crmse']:
        try:
            ft_evaluation_metrics[metric].append(ft_historys[-2][metric][-1])
        except:
            pass

    with open(os.path.join(task_save_folder, f'ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)

    torch.save(target_classifier.encoder.state_dict(), os.path.join('model_save', 'reference_encoder.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLEIT reference encoder build')

    parser.add_argument('--metric', dest='metric', nargs='?', default='cpearsonr', choices=['cpearsonr', 'dpearsonr'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    args = parser.parse_args()

    build_encoder(args)
