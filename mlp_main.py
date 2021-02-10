from evaluation_utils import evaluate_target_regression_epoch, model_save_check
from collections import defaultdict
from itertools import chain
from mlp import MLP
from mask_mlp import MaskMLP
from encoder_decoder import EncoderDecoder
from loss_and_metrics import masked_mse, masked_simse
from vae import VAE
from data import DataProvider
import torch
import json
import os
import argparse
import itertools

def regression_train_step(model, batch, device, optimizer, history, scheduler=None, clip=None):
    # gc.collect()
    # torch.cuda.empty_cache()

    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    loss = masked_simse(preds=model(x), labels=y)
    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())

    return history


def fine_tune_encoder(train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                      metric_name='dpearsonr',
                      normalize_flag=False, **kwargs):

    autoencoder = VAE(input_dim=kwargs['input_dim'],
                      latent_dim=kwargs['latent_dim'],
                      hidden_dims=kwargs['encoder_hidden_dims'],
                      dop=kwargs['dop']).to(kwargs['device'])
    encoder = autoencoder.encoder

    target_decoder = MaskMLP(input_dim=kwargs['latent_dim'],
                             output_dim=kwargs['output_dim'],
                             hidden_dims=kwargs['regressor_hidden_dims']).to(kwargs['device'])

    target_regressor = EncoderDecoder(encoder=encoder,
                                      decoder=target_decoder,
                                      normalize_flag=normalize_flag).to(kwargs['device'])

    target_regression_train_history = defaultdict(list)
    target_regression_eval_train_history = defaultdict(list)
    target_regression_eval_val_history = defaultdict(list)
    target_regression_eval_test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                              if str(list(encoder.modules())[i]).startswith('Linear')]

    reset_count = 1
    lr = kwargs['lr']

    target_regression_params = [target_regressor.decoder.parameters()]
    target_regression_optimizer = torch.optim.AdamW(chain(*target_regression_params),
                                                    lr=lr)

    for epoch in range(kwargs['train_num_epochs']):
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            target_regression_train_history = regression_train_step(model=target_regressor,
                                                                    batch=batch,
                                                                    device=kwargs['device'],
                                                                    optimizer=target_regression_optimizer,
                                                                    history=target_regression_train_history)
        target_regression_eval_train_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                dataloader=train_dataloader,
                                                                                device=kwargs['device'],
                                                                                history=target_regression_eval_train_history)
        target_regression_eval_val_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                              dataloader=val_dataloader,
                                                                              device=kwargs['device'],
                                                                              history=target_regression_eval_val_history)

        if test_dataloader is not None:
            target_regression_eval_test_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                   dataloader=test_dataloader,
                                                                                   device=kwargs['device'],
                                                                                   history=target_regression_eval_test_history)
        save_flag, stop_flag = model_save_check(history=target_regression_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=10,
                                                reset_count=reset_count)
        if save_flag:
            torch.save(target_regressor.state_dict(),
                       os.path.join(task_save_folder, f'target_regressor_{seed}.pt'))
        if stop_flag:
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing {epoch}')
                target_regressor.load_state_dict(
                    torch.load(os.path.join(task_save_folder, f'target_regressor_{seed}.pt')))

                target_regression_params.append(list(target_regressor.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_regression_optimizer = torch.optim.AdamW(chain(*target_regression_params), lr=lr)
                reset_count += 1
            except IndexError:
                break

    target_regressor.load_state_dict(
        torch.load(os.path.join(task_save_folder, f'target_regressor_{seed}.pt')))

    return target_regressor, (target_regression_train_history, target_regression_eval_train_history,
                              target_regression_eval_val_history, target_regression_eval_test_history)


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('train_params.json', 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    #patching
    training_params['labeled']['train_num_epochs'] = update_params_dict['ftrain_num_epochs']

    f_epoch = update_params_dict.pop('ftrain_num_epochs')
    param_str = dict_to_str(update_params_dict)

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', 'mlp', args.omics, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag
        })
    task_save_folder = os.path.join('model_save', 'mlp', args.omics, param_str)
    safe_make_dir(training_params['model_save_folder'])
    safe_make_dir(task_save_folder)

    data_provider = DataProvider(batch_size=training_params['unlabeled']['batch_size'],
                                 target=args.measurement)

    training_params.update(
        {
            'input_dim': data_provider.shape_dict[args.omics],
            'output_dim': data_provider.shape_dict['target']
        }
    )

    ft_evaluation_metrics = defaultdict(list)
    if args.omics == 'gex':
        labeled_dataloader_generator = data_provider.get_drug_labeled_gex_dataloader()
        fold_count = 0
        for train_labeled_dataloader, val_labeled_dataloader in labeled_dataloader_generator:
            target_regressor, ft_historys = fine_tune_encoder(
                train_dataloader=train_labeled_dataloader,
                val_dataloader=val_labeled_dataloader,
                test_dataloader=val_labeled_dataloader,
                seed=fold_count,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'drmse', 'cpearsonr', 'crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-1][metric][-1])
            fold_count += 1
    else:
        labeled_dataloader_generator = data_provider.get_drug_labeled_mut_dataloader()
        fold_count = 0
        test_ft_evaluation_metrics = defaultdict(list)

        for train_labeled_dataloader, val_labeled_dataloader, test_labeled_dataloader in labeled_dataloader_generator:
            target_regressor, ft_historys = fine_tune_encoder(
                train_dataloader=train_labeled_dataloader,
                val_dataloader=val_labeled_dataloader,
                test_dataloader=test_labeled_dataloader,
                seed=fold_count,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'drmse', 'cpearsonr', 'crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-2][metric][-1])
                test_ft_evaluation_metrics[metric].append(ft_historys[-1][metric][-1])
            fold_count += 1
        with open(os.path.join(task_save_folder, f'{param_str}_test_ft_evaluation_results.json'), 'w') as f:
            json.dump(test_ft_evaluation_metrics, f)
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CLEIT training and evaluation')
    parser.add_argument('--omics', dest='omics', nargs='?', default='gex',
                        choices=['gex', 'mut'])
    # parser.add_argument('--drug', dest='drug', nargs='?', default='gem', choices=['gem', 'fu', 'cis', 'tem'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='cpearsonr', choices=['cpearsonr', 'dpearsonr'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    args = parser.parse_args()

    params_grid = {
        "train_num_epochs": [1000],
        "dop": [0.0, 0.1],
        "ftrain_num_epochs": [100]
    }

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param_dict in update_params_dict_list:
        main(args=args, update_params_dict=param_dict)
