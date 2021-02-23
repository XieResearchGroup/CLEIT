import torch
import torch.nn as nn
import os
from collections import defaultdict
from ae import AE
from mlp import MLP
from itertools import chain
from evaluation_utils import model_save_check, evaluate_target_regression_epoch
from loss_and_metrics import masked_simse, masked_mse
from multi_out_mlp import MoMLP
from encoder_decoder import EncoderDecoder


def dann_train_step(classifier, model, s_batch, t_batch, loss_fn, device, optimizer, alpha, history,
                    scheduler=None,
                    clip=None):
    classifier.zero_grad()
    model.zero_grad()
    classifier.train()
    model.train()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)

    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    outputs = torch.cat((classifier(s_x), classifier(t_x)), dim=0)
    truths = torch.cat((torch.zeros(s_x.shape[0], 1), torch.ones(t_x.shape[0], 1)), dim=0).to(device)
    dann_loss = loss_fn(outputs, truths)
    loss = masked_mse(preds=model(s_x), labels=s_y) + masked_mse(preds=model(t_x), labels=t_y) + alpha * dann_loss

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    if clip is not None:
        for p in classifier.decoder.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())
    history['bce'].append(dann_loss.cpu().detach().item())

    return history


def train_dann(s_dataloaders, t_dataloaders, val_dataloader, test_dataloader, metric_name, seed, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders
    t_train_dataloader = t_dataloaders

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
                                      decoder=target_decoder).to(kwargs['device'])

    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=1,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop'],
                     out_fn=torch.nn.Sigmoid).to(kwargs['device'])

    confounder_classifier = EncoderDecoder(encoder=autoencoder.encoder,
                                           decoder=classifier).to(kwargs['device'])

    train_history = defaultdict(list)
    s_target_regression_eval_train_history = defaultdict(list)
    t_target_regression_eval_train_history = defaultdict(list)
    target_regression_eval_val_history = defaultdict(list)
    target_regression_eval_test_history = defaultdict(list)

    confounded_loss = nn.BCEWithLogitsLoss()
    dann_params = [target_regressor.parameters(),
                   confounder_classifier.decoder.parameters()]
    dann_optimizer = torch.optim.AdamW(chain(*dann_params), lr=kwargs['lr'])

    # start alternative training
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'DANN training epoch {epoch}')
        # start autoencoder training epoch
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            train_history = dann_train_step(classifier=confounder_classifier,
                                            model=target_regressor,
                                            s_batch=s_batch,
                                            t_batch=t_batch,
                                            loss_fn=confounded_loss,
                                            alpha=kwargs['alpha'],
                                            device=kwargs['device'],
                                            optimizer=dann_optimizer,
                                            history=train_history,
                                            scheduler=None)

        s_target_regression_eval_train_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                  dataloader=s_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=s_target_regression_eval_train_history)

        t_target_regression_eval_train_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                                  dataloader=t_train_dataloader,
                                                                                  device=kwargs['device'],
                                                                                  history=t_target_regression_eval_train_history)
        target_regression_eval_val_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                              dataloader=val_dataloader,
                                                                              device=kwargs['device'],
                                                                              history=target_regression_eval_val_history)
        target_regression_eval_test_history = evaluate_target_regression_epoch(regressor=target_regressor,
                                                                               dataloader=test_dataloader,
                                                                               device=kwargs['device'],
                                                                               history=target_regression_eval_test_history)

        save_flag, stop_flag = model_save_check(history=target_regression_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=50)
        if save_flag:
            torch.save(target_regressor.state_dict(), os.path.join(kwargs['model_save_folder'], f'dann_regressor_{seed}.pt'))
        if stop_flag:
            break
    target_regressor.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], f'dann_regressor_{seed}.pt')))

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

    return target_regressor, (
        train_history, s_target_regression_eval_train_history, t_target_regression_eval_train_history,
        target_regression_eval_val_history, target_regression_eval_test_history)
