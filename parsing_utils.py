import re
import json
import os
import pickle
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import Counter, defaultdict


def get_largest_kv(d, std_dict):
    k = max(d.items(), key=itemgetter(1))[0]
    return k, d[k], std_dict[k]


def parse_param_str(param_str):
    pattern = re.compile('(pretrain_num_epochs)?_?(\d+)?_?(train_num_epochs)_(\d+)_(dop)_(\d\.\d)')
    matches = pattern.findall(param_str)
    return {matches[0][i]: float(matches[0][i + 1]) for i in range(0, len(matches[0]), 2) if matches[0][i] != ''}

def parse_hyper_vae_ft_evaluation_result(metric_name='cpearsonr'):
    folder = 'model_save/vae/gex/'
    evaluation_metrics = {}
    evaluation_metrics_std = {}
    file_pattern = '(pretrain|train)+.*(dop+).*(ft)+.*\.json'

    for sub_folder in os.listdir(folder):
        if re.match('(pretrain|train)+.*(dop+).*', sub_folder):
            for file in os.listdir(os.path.join(folder, sub_folder)):
                if re.match(file_pattern, file):
                    with open(os.path.join(folder, sub_folder,file), 'r') as f:
                        result_dict = json.load(f)
                    metrics = result_dict[metric_name]
                    if any(np.isnan(metrics)):
                        pass
                    else:
                        evaluation_metrics[file] = np.mean(result_dict[metric_name])
                        evaluation_metrics_std[file] = np.std(result_dict[metric_name])

    print(evaluation_metrics)
    # print(evaluation_metrics_std)
    print(get_largest_kv(d=evaluation_metrics, std_dict=evaluation_metrics_std))
    return parse_param_str(get_largest_kv(d=evaluation_metrics, std_dict=evaluation_metrics_std)[0])

def parse_ft_param_str(param_str):
    ftrain_num_epochs = int(param_str[:param_str.find('_')])
    param_str = param_str[param_str.find('_') + 1:]
    pattern = re.compile('(pretrain_num_epochs)?_?(\d+)?_?(train_num_epochs)_(\d+)_(dop)_(\d\.\d)')
    matches = pattern.findall(param_str)
    return {matches[0][i]: float(matches[0][i + 1]) for i in range(0, len(matches[0]), 2) if
            matches[0][i] != ''}, ftrain_num_epochs


# def parse_hyper_vae_ft_evaluation_result(metric_name='dpearsonr'):
#     folder = 'model_save/vae/gex/'
#     evaluation_metrics = {}
#     evaluation_metrics_std = {}
#     for sub_folder in os.listdir(folder):
#         if re.match('(pretrain|train)+.*(dop+).*', sub_folder):
#             for d in os.listdir(os.path.join(folder, sub_folder)):
#                 if re.match('(ftrain)+', d):
#                     ft_train_epoch = d.split('_')[-1]
#                     for file in os.listdir(os.path.join(folder, sub_folder, d)):
#                         if file.endswith('json'):
#                             with open(os.path.join(folder, sub_folder, d, file), 'r') as f:
#                                 result_dict = json.load(f)
#                             metrics = result_dict[metric_name]
#                             if any(np.isnan(metrics)):
#                                 pass
#                             else:
#                                 evaluation_metrics["_".join([ft_train_epoch, file])] = np.mean(metrics)
#                                 evaluation_metrics_std["_".join([ft_train_epoch, file])] = np.std(metrics)
#     print(evaluation_metrics)
#     # print(evaluation_metrics_std)
#     print(get_largest_kv(d=evaluation_metrics, std_dict=evaluation_metrics_std))
#     return parse_ft_param_str(get_largest_kv(d=evaluation_metrics, std_dict=evaluation_metrics_std)[0])


def parse_ft_evaluation_result(file_name, method, measurement='AUC', metric_name='cpearsonr'):
    folder = f'model_save/{method}/{measurement}'
    with open(os.path.join(folder, file_name), 'r') as f:
        result_dict = json.load(f)
    return result_dict[metric_name]


def parse_hyper_ft_evaluation_result(method, measurement='AUC', metric_name='cpearsonr', test_flag=True):
    folder = f'model_save/{method}/{measurement}'
    evaluation_metrics = {}
    evaluation_metrics_std = {}
    count = 0
    file_pattern = '(pretrain|train)+.*(dop+).*(test_ft)+.*\.json' if test_flag else '(pretrain|train)+.*(dop+).*(ft)+.*\.json'
    for file in os.listdir(folder):
        if re.match(file_pattern, file):
            count += 1
            with open(os.path.join(folder, file), 'r') as f:
                result_dict = json.load(f)
            metrics = result_dict[metric_name]
            if sum(np.isnan(metrics)) > 4:
                pass
            else:
                evaluation_metrics[file] = np.nanmean(result_dict[metric_name])
                evaluation_metrics_std[file] = np.nanstd(result_dict[metric_name])

    return evaluation_metrics, evaluation_metrics_std, count


def generate_hyper_ft_report(metric_name='cpearsonr', measurement='AUC'):
    methods = ['mlp', 'adda', 'dann', 'dcc', 'coral', 'dsn', 'cleit', 'cleitc', 'cleita', 'cleitm']
    report = pd.DataFrame(np.zeros((len(methods), 2)), index=methods, columns=['mean', 'std'])
    result_dict = defaultdict(dict)

    for method in methods:
        if method in ['mlp','adda','dann','dcc','coral']:
            if method == 'mlp':
                folder = f'model_save/{method}/mut'
            else:
                folder = f'model_save/{method}/labeled'
            with open(os.path.join(folder, 'test_ft_evaluation_results.json'), 'r') as f:
                result_dict[method] = json.load(f)[metric_name]
                report.loc[method, 'mean'] = np.nanmean(result_dict[method])
                report.loc[method, 'std'] = np.nanstd(result_dict[method])
        else:
            #folder = f'model_save/{method}'
            try:
                param_str, report.loc[method, 'mean'], report.loc[method, 'std'] = get_largest_kv(d=
                                                                                                  parse_hyper_ft_evaluation_result(
                                                                                                      method=method,
                                                                                                      metric_name=metric_name,
                                                                                                      measurement=measurement)[
                                                                                                      0],
                                                                                                  std_dict=
                                                                                                  parse_hyper_ft_evaluation_result(
                                                                                                      method=method,
                                                                                                      metric_name=metric_name,
                                                                                                      measurement=measurement)[
                                                                                                      1])

                result_dict[method] = parse_ft_evaluation_result(file_name=param_str, method=method,
                                                                 metric_name=metric_name, measurement=measurement)
                # with open(os.path.join(folder, 'train_params.json'), 'r') as f:
                #     params_dict = json.load(f)
                # params_dict['unlabeled'].update(parse_param_str(param_str))
                # with open(os.path.join(folder, f'train_params.json'), 'w') as f:
                #     json.dump(params_dict, f)
            except:
                pass
    return report, result_dict


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data
