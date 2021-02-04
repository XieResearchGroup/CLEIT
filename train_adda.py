import os
import torch.autograd as autograd
from ae import AE
from evaluation_utils import *
from mlp import MLP


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


def critic_train_step(critic, ae, s_batch, t_batch, device, optimizer, history, scheduler=None,
                      clip=None, gp=None):
    critic.zero_grad()
    ae.zero_grad()
    ae.eval()
    critic.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_code = ae.encode(s_x)
    t_code = ae.encode(t_x)

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


def gan_gen_train_step(critic, ae, s_batch, t_batch, device, optimizer, alpha, history,
                       scheduler=None):
    critic.zero_grad()
    ae.zero_grad()
    critic.eval()
    ae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    t_code = ae.encode(t_x)

    optimizer.zero_grad()
    gen_loss = -torch.mean(critic(t_code))
    s_loss_dict = ae.loss_function(*ae(s_x))
    t_loss_dict = ae.loss_function(*ae(t_x))
    recons_loss = s_loss_dict['loss'] + t_loss_dict['loss']
    loss = recons_loss + alpha * gen_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['gen_loss'].append(gen_loss.cpu().detach().item())

    return history


def train_adda(s_dataloaders, t_dataloaders, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders
    s_test_dataloader = s_dataloaders

    t_train_dataloader = t_dataloaders
    t_test_dataloader = t_dataloaders

    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     noise_flag=False,
                     dop=kwargs['dop']).to(kwargs['device'])

    confounding_classifier = MLP(input_dim=kwargs['latent_dim'],
                                 output_dim=1,
                                 hidden_dims=kwargs['classifier_hidden_dims'],
                                 dop=kwargs['dop']).to(kwargs['device'])

    ae_train_history = defaultdict(list)
    ae_val_history = defaultdict(list)
    critic_train_history = defaultdict(list)
    gen_train_history = defaultdict(list)

    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])
        classifier_optimizer = torch.optim.RMSprop(confounding_classifier.parameters(), lr=kwargs['lr'])
        for epoch in range(int(kwargs['train_num_epochs'])):
            if epoch % 50 == 0:
                print(f'confounder wgan training epoch {epoch}')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                critic_train_history = critic_train_step(critic=confounding_classifier,
                                                         ae=autoencoder,
                                                         s_batch=s_batch,
                                                         t_batch=t_batch,
                                                         device=kwargs['device'],
                                                         optimizer=classifier_optimizer,
                                                         history=critic_train_history,
                                                         # clip=0.1,
                                                         gp=10.0)
                if (step + 1) % 5 == 0:
                    gen_train_history = gan_gen_train_step(critic=confounding_classifier,
                                                           ae=autoencoder,
                                                           s_batch=s_batch,
                                                           t_batch=t_batch,
                                                           device=kwargs['device'],
                                                           optimizer=ae_optimizer,
                                                           alpha=1.0,
                                                           history=gen_train_history)

        torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'ae.pt'))
    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_train_history, ae_val_history, critic_train_history, gen_train_history)
