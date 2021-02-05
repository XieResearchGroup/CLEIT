import torch
import os
import torch.autograd as autograd
from collections import defaultdict
from itertools import chain
from vae import VAE
from evaluation_utils import *
from mlp import MLP
from encoder_decoder import EncoderDecoder
from copy import deepcopy


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


def critic_train_step(critic, ae, reference_encoder, transmitter, batch, device, optimizer, history, scheduler=None,
                          clip=None, gp=None):
    critic.zero_grad()
    ae.zero_grad()
    reference_encoder.zero_grad()
    transmitter.zero_grad()
    ae.eval()
    transmitter.eval()
    reference_encoder.eval()
    critic.train()

    x_m = batch[0].to(device)
    x_g = batch[1].to(device)

    x_m_code = transmitter(ae.encoder(x_m))
    x_g_code = reference_encoder(x_g)

    loss = torch.mean(critic(x_m_code)) - torch.mean(critic(x_g_code))
    if gp is not None:
        gradient_penalty = compute_gradient_penalty(critic,
                                                    real_samples=x_g_code,
                                                    fake_samples=x_m_code,
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


def gan_gen_train_step(critic, ae, transmitter, batch, device, optimizer, alpha, history,
                           scheduler=None):
    critic.zero_grad()
    ae.zero_grad()
    transmitter.zero_grad()
    ae.train()
    transmitter.train()
    critic.eval()

    x_m = batch[0].to(device)
    x_m_code = transmitter(ae.encoder(x_m))

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(x_m_code))

    loss_dict = ae.loss_function(*ae(x_m))
    loss = loss_dict['loss'] + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_cleita(dataloader, **kwargs):
    autoencoder = VAE(input_dim=kwargs['input_dim'],
                      latent_dim=kwargs['latent_dim'],
                      hidden_dims=kwargs['encoder_hidden_dims'],
                      dop=kwargs['dop']).to(kwargs['device'])

    # get reference encoder
    aux_ae = deepcopy(autoencoder)

    aux_ae.encoder.load_state_dict(torch.load(os.path.join('./model_save', 'reference_encoder.pt')))
    reference_encoder = aux_ae.encoder

    # construct transmitter
    transmitter = MLP(input_dim=kwargs['latent_dim'],
                      output_dim=kwargs['latent_dim'],
                      hidden_dims=[kwargs['latent_dim']]).to(kwargs['device'])

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'],
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    ae_train_history = defaultdict(list)
    ae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)

    if kwargs['retrain_flag']:
        cleit_params = [
            autoencoder.parameters(),
            transmitter.parameters()
        ]
        cleit_optimizer = torch.optim.AdamW(chain(*cleit_params), lr=kwargs['lr'])
        classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for step, batch in enumerate(dataloader):
                critic_train_history = critic_train_step(critic=confounding_classifier,
                                                             ae=autoencoder,
                                                             reference_encoder=reference_encoder,
                                                             transmitter=transmitter,
                                                             batch=batch,
                                                             device=kwargs['device'],
                                                             optimizer=classifier_optimizer,
                                                             history=critic_train_history,
                                                             # clip=0.1,
                                                             gp=10.0)
                if (step + 1) % 5 == 0:
                    gen_train_history = gan_gen_train_step(critic=confounding_classifier,
                                                               ae=autoencoder,
                                                               transmitter=transmitter,
                                                               batch=batch,
                                                               device=kwargs['device'],
                                                               optimizer=cleit_optimizer,
                                                               alpha=1.0,
                                                               history=gen_train_history)

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'cleit_vae.pt'))
        torch.save(transmitter.state_dict(), os.path.join(kwargs['model_save_folder'], 'transmitter.pt'))
    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cleit_vae.pt')))
            transmitter.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'transmitter.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    encoder = EncoderDecoder(encoder=autoencoder.encoder,
                             decoder=transmitter).to(kwargs['device'])

    return encoder, (ae_train_history, ae_val_history, critic_train_history, gen_train_history)
