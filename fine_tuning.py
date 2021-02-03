from evaluation_utils import evaluate_target_regression_epoch, model_save_check
from collections import defaultdict
from itertools import chain
from mlp import MLP
from mask_mlp import MaskMLP
from encoder_decoder import EncoderDecoder
from torch.nn import functional as F

import os
import torch


def classification_train_step(model, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    loss = loss_fn(model(x), y.double().unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history


def regression_train_step(model, batch, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    mse_loss = sum([F.mse_loss(torch.where(torch.isnan(y[i, :]), torch.zeros_like(y[i, :]), y[i, :]),
                               torch.where(torch.isnan(y[i, :]), torch.zeros_like(y[i, :]), model(x)[i, :]))
                    for i in range(y.shape[0])])
    penalty_term = sum([torch.square(torch.sum(
        torch.where(torch.isnan(y[i, :]), torch.zeros_like(y[i, :]), y[i, :]) -
        torch.where(torch.isnan(y[i, :]), torch.zeros_like(y[i, :]), model(x)[i, :]))) / torch.square(
        (~torch.isnan(y[i, :])).sum())
                        for i in range(y.shape[0])])

    loss = (mse_loss - penalty_term) / y.shape[0]

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())

    return history


def fine_tune_encoder(encoder, train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                      metric_name='dpearsonr',
                      normalize_flag=False, **kwargs):
    # target_decoder = MLP(input_dim=kwargs['latent_dim'],
    #                      output_dim=1,
    #                      hidden_dims=kwargs['regressor_hidden_dims']).to(kwargs['device'])
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


def fine_tune_encoder_new(encoder, train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                          metric_name='dpearsonr',
                          normalize_flag=False, **kwargs):
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

    lr = kwargs['lr']

    target_regression_params = [target_regressor.decoder.parameters()]
    target_regression_optimizer = torch.optim.AdamW(chain(*target_regression_params),
                                                    lr=lr)
    gu_flag = True
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

        if epoch >= 0.5 * kwargs['train_num_epochs'] and epoch % 10 == 0 and gu_flag:
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing {epoch}')
                target_regression_params.append(list(target_regressor.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_regression_optimizer = torch.optim.AdamW(chain(*target_regression_params), lr=lr)
            except IndexError:
                gu_flag = False

    target_regressor.load_state_dict(
        torch.load(os.path.join(task_save_folder, f'target_regressor_{seed}.pt')))

    return target_regressor, (target_regression_train_history, target_regression_eval_train_history,
                              target_regression_eval_val_history, target_regression_eval_test_history)
