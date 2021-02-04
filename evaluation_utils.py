import numpy as np
import numpy.ma as ma
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss, auc, precision_recall_curve
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def model_save_check(history, metric_name, tolerance_count=5, reset_count=1):
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0
        save_flag = True
    if metric_name.endswith('loss') or metric_name.endswith('mse'):
        if history[metric_name][-1] <= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    if len(history[metric_name]) - history['best_index'] > tolerance_count * reset_count and history['best_index'] > 0:
        stop_flag = True

    return save_flag, stop_flag


def eval_ae_epoch(model, data_loader, device, history):
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history


def evaluate_target_classification_epoch(classifier, dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
            y_pred = torch.sigmoid(classifier(x_batch)).detach()
            y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history


def evaluate_target_regression_epoch(regressor, dataloader, device, history):
    y_truths = None
    y_preds = None
    regressor.eval()

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_truths = np.vstack(
                [y_truths, y_batch.cpu().detach().numpy()]) if y_truths is not None else y_batch.cpu().detach().numpy()
            y_pred = regressor(x_batch).detach()
            y_preds = np.vstack([y_preds,
                                 y_pred.cpu().detach().numpy()]) if y_preds is not None else y_pred.cpu().detach().numpy()
    assert (y_truths.shape == y_preds.shape)
    history['dpearsonr'].append(np.mean([pearsonr(y_truths[:, i][~ma.masked_invalid(y_truths[:, i]).mask],
                                                  y_preds[:, i][~ma.masked_invalid(y_truths[:, i]).mask])[0] for i in
                                         range(y_truths.shape[1])]).item())
    history['cpearsonr'].append(np.mean([pearsonr(y_truths[i, :][~ma.masked_invalid(y_truths[i, :]).mask],
                                                  y_preds[i, :][~ma.masked_invalid(y_truths[i, :]).mask])[0] for i in
                                         range(y_truths.shape[0])]).item())
    # history['dspearmanr'].append(np.mean([spearmanr(y_truths[:, i][~ma.masked_invalid(y_truths[:, i]).mask],
    #                                                 y_preds[:, i][~ma.masked_invalid(y_truths[:, i]).mask])[0] for i in
    #                                       range(y_truths.shape[1])]).item())
    # history['cspearmanr'].append(np.mean([spearmanr(y_truths[i, :][~ma.masked_invalid(y_truths[i, :]).mask],
    #                                                 y_preds[i, :][~ma.masked_invalid(y_truths[i, :]).mask])[0] for i in
    #                                       range(y_truths.shape[0])]).item())
    history['drmse'].append(np.mean([mean_squared_error(y_truths[:, i][~ma.masked_invalid(y_truths[:, i]).mask],
                                                        y_preds[:, i][~ma.masked_invalid(y_truths[:, i]).mask],
                                                        squared=False) for i in range(y_truths.shape[1])]).item())
    history['crmse'].append(np.mean([mean_squared_error(y_truths[i, :][~ma.masked_invalid(y_truths[i, :]).mask],
                                                        y_preds[i, :][~ma.masked_invalid(y_truths[i, :]).mask],
                                                        squared=False) for i in range(y_truths.shape[0])]).item())

    # history['pearsonr'].append(pearsonr(y_truths, y_preds)[0])
    # history['spearmanr'].append(spearmanr(y_truths, y_preds)[0])
    # history['r2'].append(r2_score(y_true=y_truths, y_pred=y_preds))
    # history['rmse'].append(mean_squared_error(y_true=y_truths, y_pred=y_preds, squared=False))
    return history


def evaluate_adv_classification_epoch(classifier, s_dataloader, t_dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for s_batch in s_dataloader:
        s_x = s_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.zeros(s_x.shape[0]).ravel()])
            s_y_pred = torch.sigmoid(classifier(s_x)).detach()
            y_preds = np.concatenate([y_preds, s_y_pred.cpu().detach().numpy().ravel()])

    for t_batch in t_dataloader:
        t_x = t_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.ones(t_x.shape[0]).ravel()])
            t_y_pred = torch.sigmoid(classifier(t_x)).detach()
            y_preds = np.concatenate([y_preds, t_y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history
