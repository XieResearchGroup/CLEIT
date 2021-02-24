import os
import torch.autograd as autograd
from ae import AE
from mlp import MLP
import torch
from collections import defaultdict
from evaluation_utils import model_save_check, evaluate_target_regression_epoch
from loss_and_metrics import masked_simse, masked_mse
from multi_out_mlp import MoMLP
from encoder_decoder import EncoderDecoder


def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.shape[0], 1)).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    fakes = torch.ones((real_samples.shape[0], 1)).to(device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=fakes,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def critic_train_step(critic, model, s_batch, t_batch, device, optimizer, history, scheduler=None,
                      clip=None, gp=None):
    critic.zero_grad()
    model.zero_grad()
    model.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = model.encode(s_x)
    t_code = model.encode(t_x)

    loss = torch.mean(critic(t_code)) - torch.mean(critic(s_code))
    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=s_code,
                                                    fake_samples=t_code,
                                                    device=device)
        loss = loss + gp * gradient_penalty

    optimizer.zero_grad()
    loss.backward()
    #     if clip is not None:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    if clip is not None:
        for p in critic.parameters():
            p.data.clamp_(-clip, clip)
    if scheduler is not None:
        scheduler.step()

    history['critic_loss'].append(loss.cpu().detach().item())

    return history


def gan_gen_train_step(critic, model, s_batch, t_batch, device, optimizer, alpha, history,
                       scheduler=None):
    critic.zero_grad()
    model.zero_grad()
    critic.eval()
    model.train()

    s_x = s_batch[0].to(device)
    s_y = s_batch[1].to(device)

    t_x = t_batch[0].to(device)
    t_y = t_batch[1].to(device)

    t_code = model.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))

    loss = masked_mse(preds=model(s_x), labels=s_y) + masked_mse(preds=model(t_x), labels=t_y) + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['loss'].append(loss.cpu().detach().item())
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_adda(s_dataloaders, t_dataloaders, val_dataloader, test_dataloader, metric_name, seed, **kwargs):
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

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'],
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)
    s_target_regression_eval_train_history = defaultdict(list)
    t_target_regression_eval_train_history = defaultdict(list)
    target_regression_eval_val_history = defaultdict(list)
    target_regression_eval_test_history = defaultdict(list)

    model_optimizer = torch.optim.AdamW(target_regressor.parameters(), lr=kwargs['lr'])
    classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
    for epoch in range(int(kwargs['train_num_epochs'])):
        if epoch % 50 == 0:
            print(f'ADDA training epoch {epoch}')
        for step, s_batch in enumerate(s_train_dataloader):
            t_batch = next(iter(t_train_dataloader))
            critic_train_history = critic_train_step(critic=confounding_classifier,
                                                     model=target_regressor,
                                                     s_batch=s_batch,
                                                     t_batch=t_batch,
                                                     device=kwargs['device'],
                                                     optimizer=classifier_optimizer,
                                                     history=critic_train_history,
                                                     # clip=0.1,
                                                     gp=10.0)
            if (step + 1) % 5 == 0:
                gen_train_history = gan_gen_train_step(critic=confounding_classifier,
                                                       model=target_regressor,
                                                       s_batch=s_batch,
                                                       t_batch=t_batch,
                                                       device=kwargs['device'],
                                                       optimizer=model_optimizer,
                                                       alpha=1.0,
                                                       history=gen_train_history)
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
            torch.save(target_regressor.state_dict(),
                       os.path.join(kwargs['model_save_folder'], f'adda_regressor_{seed}.pt'))
        if stop_flag:
            break

    target_regressor.load_state_dict(
        torch.load(os.path.join(kwargs['model_save_folder'], f'adda_regressor_{seed}.pt')))

    # evaluate_target_regression_epoch(regressor=target_regressor,
    #                                  dataloader=val_dataloader,
    #                                  device=kwargs['device'],
    #                                  history=None,
    #                                  seed=seed,
    #                                  output_folder=kwargs['model_save_folder'])
    evaluate_target_regression_epoch(regressor=target_regressor,
                                     dataloader=test_dataloader,
                                     device=kwargs['device'],
                                     history=None,
                                     seed=seed,
                                     output_folder=kwargs['model_save_folder'])

    return target_regressor, (
        critic_train_history, gen_train_history, s_target_regression_eval_train_history,
        t_target_regression_eval_train_history,
        target_regression_eval_val_history, target_regression_eval_test_history)
