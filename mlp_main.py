from evaluation_utils import evaluate_target_regression_epoch, model_save_check
from collections import defaultdict
from mlp import MLP
from multi_out_mlp import MoMLP
from encoder_decoder import EncoderDecoder
from loss_and_metrics import masked_mse, masked_simse
from ae import AE
from data import DataProvider
import torch
import json
import os
import argparse


def regression_train_step(model, batch, device, optimizer, history, scheduler=None, clip=None):
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


def fine_tune_encoder(train_dataloader, val_dataloader, seed, test_dataloader=None,
                      metric_name='cpearsonr',
                      normalize_flag=False, **kwargs):
    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    encoder = autoencoder.encoder

    target_decoder = MoMLP(input_dim=kwargs['latent_dim'],
                           output_dim=kwargs['output_dim'],
                           hidden_dims=kwargs['regressor_hidden_dims'],
                           out_fn=torch.nn.Sigmoid).to(kwargs['device'])

    target_regressor = EncoderDecoder(encoder=encoder,
                                      decoder=target_decoder,
                                      normalize_flag=normalize_flag).to(kwargs['device'])

    target_regression_train_history = defaultdict(list)
    target_regression_eval_train_history = defaultdict(list)
    target_regression_eval_val_history = defaultdict(list)
    target_regression_eval_test_history = defaultdict(list)

    target_regression_optimizer = torch.optim.AdamW(target_regressor.parameters(), lr=kwargs['lr'])

    for epoch in range(kwargs['train_num_epochs']):
        if epoch % 50 == 0:
            print(f'MLP fine-tuning epoch {epoch}')
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
                                                tolerance_count=50)
        if save_flag:
            torch.save(target_regressor.state_dict(),
                       os.path.join(kwargs['model_save_folder'], f'target_regressor_{seed}.pt'))
        if stop_flag:
            break

    target_regressor.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], f'target_regressor_{seed}.pt')))

    evaluate_target_regression_epoch(regressor=target_regressor,
                                     dataloader=val_dataloader,
                                     device=kwargs['device'],
                                     history=None,
                                     seed=seed,
                                     output_folder=kwargs['model_save_folder'])
    evaluate_target_regression_epoch(regressor=target_regressor,
                                     dataloader=test_dataloader,
                                     device=kwargs['device'],
                                     history=None,
                                     seed=seed,
                                     output_folder=kwargs['model_save_folder'])

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


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join('model_save', 'train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params.update(
        {
            'device': device,
            'model_save_folder': os.path.join('model_save', 'mlp', args.omics),
        })

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
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'dspearmanr','drmse', 'cpearsonr', 'cspearmanr','crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-2][metric][ft_historys[-2]['best_index']])
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
                **wrap_training_params(training_params, type='labeled')
            )
            for metric in ['dpearsonr', 'dspearmanr','drmse', 'cpearsonr', 'cspearmanr','crmse']:
                ft_evaluation_metrics[metric].append(ft_historys[-2][metric][ft_historys[-2]['best_index']])
                test_ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
            fold_count += 1
        with open(os.path.join(training_params['model_save_folder'], f'test_ft_evaluation_results.json'), 'w') as f:
            json.dump(test_ft_evaluation_metrics, f)
    with open(os.path.join(training_params['model_save_folder'], f'ft_evaluation_results.json'), 'w') as f:
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
    parser.set_defaults(retrain_flag=True)

    args = parser.parse_args()

    main(args=args)
