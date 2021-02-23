from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score, precision_recall_curve, accuracy_score, \
    f1_score, auc, mean_squared_error
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNet, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
from functools import partial
from data import DataProvider

import random
import json
import numpy as np
import numpy.ma as ma
import pandas as pd


def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def pearson(y_true, y_pred):
    r_val, p_val = pearsonr(x=y_true, y=y_pred)
    return r_val


def spearman(y_true, y_pred):
    r_val, p_val = spearmanr(a=y_true, b=y_pred)
    return r_val


classify_scoring = {
    'auroc': 'roc_auc',
    'auprc': make_scorer(auprc, needs_proba=True),
    'acc': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score),
    'aps': make_scorer(average_precision_score, needs_proba=True)
}

regress_scoring = {
    'mse': make_scorer(mean_squared_error),
    'pearsonr': make_scorer(pearson),
    'spearmanr': make_scorer(spearman),
}


def classify_with_rf(train_features, y_train, cv_split_rf, metric='auroc'):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 50, 200, 500, 1000], 'max_depth': [10, 50, 100, 200, 500]}]
        # rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestClassifier(), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2, scoring=classify_scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf, scaler

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def regress_with_rf(train_features, y_train, cv_split_rf, metric='pearsonr'):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 50, 200, 500, 1000], 'max_depth': [10, 50, 100, 200, 500]}]
        # rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestRegressor(), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2, scoring=regress_scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf, scaler

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def classify_with_enet(train_features, y_train, cv_split_enet, metric='auroc'):
    try:
        # logger.debug("Training elastic net regression model")
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        # alphas = [0.1]
        # l1_ratios = [0.25]
        base_enet = SGDClassifier(loss='log', penalty='elasticnet', random_state=12345)
        enet_param_grid = dict(alpha=alphas, l1_ratio=l1_ratios)
        enet = GridSearchCV(estimator=base_enet, param_grid=enet_param_grid, n_jobs=-1, cv=cv_split_enet, verbose=2,
                            scoring=classify_scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net classification model successfully")
        return enet, scaler
    except Exception as e:
        raise e


def regress_with_enet(train_features, y_train, cv_split_enet, metric='pearsonr'):
    try:
        # logger.debug("Training elastic net regression model")
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        # alphas = [0.1]
        # l1_ratios = [0.25]
        # base_enet = SGDRegressor(penalty='elasticnet', random_state=12345)
        base_enet = ElasticNet(random_state=12345, max_iter=5000)
        enet_param_grid = dict(alpha=alphas, l1_ratio=l1_ratios)
        enet = GridSearchCV(estimator=base_enet, param_grid=enet_param_grid, n_jobs=-1, cv=cv_split_enet, verbose=2,
                            scoring=regress_scoring, refit=metric)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)

        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net classification model successfully")
        return enet, scaler
    except Exception as e:
        raise e


def n_time_cv_classify(train_data, n=10, model_fn=classify_with_enet, test_data=None, random_state=2020,
                       metric='auroc'):
    # metric_list = ['auroc', 'acc', 'aps', 'f1']
    metric_list = ['auroc', 'acc', 'aps', 'f1', 'auprc']

    random.seed(random_state)
    seeds = random.sample(range(100000), k=n)
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    models = []
    for seed in seeds:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cv_split = kfold.split(*train_data)
        trained_model, scaler = model_fn(*train_data, list(cv_split), metric=metric)
        for metric in metric_list:
            train_history[metric].append(trained_model.cv_results_[f'mean_test_{metric}'][trained_model.best_index_])
        if test_data is not None:
            # preds = trained_model.predict(test_data[0])
            # pred_scores = trained_model.predict_proba(test_data[0])[:, 1]
            preds = trained_model.predict(scaler.transform(test_data[0]))
            pred_scores = trained_model.predict_proba(scaler.transform(test_data[0]))[:, 1]

            # print(preds)
            # print(pred_scores)
            test_history['auroc'].append(roc_auc_score(y_true=test_data[1], y_score=pred_scores))
            test_history['acc'].append(accuracy_score(y_true=test_data[1], y_pred=preds))
            test_history['aps'].append(average_precision_score(y_true=test_data[1], y_score=pred_scores))
            test_history['f1'].append(f1_score(y_true=test_data[1], y_pred=preds))
            test_history['auprc'].append(auprc(y_true=test_data[1], y_score=pred_scores))

        models.append(trained_model)

    return (train_history, models) if test_data is None else (train_history, test_history, models)


def multi_regress(train_data, output_file_name, model_fn=regress_with_enet, test_data=None, random_state=2020):
    train_feature_df = train_data[0]
    train_target_df = train_data[1]
    train_pred_df = pd.DataFrame(np.full_like(train_target_df, fill_value=-1),
                                 index=train_target_df.index,
                                 columns=train_target_df.columns)
    if test_data is not None:
        test_feature_df = test_data[0]
        test_target_df = test_data[1]
        test_pred_df = pd.DataFrame(np.full_like(test_target_df, fill_value=-1),
                                    index=test_target_df.index,
                                    columns=test_target_df.columns)
        assert all(train_pred_df.columns == test_pred_df.columns)

    for drug in train_pred_df.columns:
        print("Training: {}".format(drug))
        y = train_target_df.loc[~train_target_df[drug].isna(), drug]
        sample_ids = train_feature_df.index.intersection(y.index)
        X = train_feature_df.loc[sample_ids]
        # print(X.columns)
        y = y.reindex(sample_ids)

        outer_kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        for train_index, test_index in outer_kfold.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_split = kfold.split(X_train)
            try:
                trained_model, scaler = model_fn(X_train, y_train, list(cv_split))
                train_pred_df.loc[y.index[test_index], drug] = trained_model.predict(scaler.transform(X_test))
            except Exception as e:
                print(e)
        assert all(train_pred_df.index==train_target_df.index)
        train_pred_df.to_csv(f'./predictions/{output_file_name}.csv', index_label='Sample')

        if test_data is not None:
            outer_kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_split = outer_kfold.split(X)
            try:
                trained_model, scaler = model_fn(X, y, list(cv_split))
                test_pred_df.loc[test_feature_df.index, drug] = trained_model.predict(scaler.transform(test_feature_df))
            except Exception as e:
                print(e)
            assert all(test_pred_df.index == test_target_df.index)
            test_pred_df.to_csv(f'./predictions/test_{output_file_name}.csv', index_label='Sample')


    return (train_target_df.values, train_pred_df.values) if test_data is None else (
        train_target_df.values, train_pred_df.values, test_target_df.values, test_pred_df.values)


def n_time_cv_regress(train_data, output_file_name, n=5, model_fn=regress_with_enet, test_data=None, random_state=2020):
    random.seed(random_state)
    seeds = random.sample(range(100000), k=int(n))
    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for seed in seeds:
        if test_data is None:
            train_y_truths, train_y_preds = multi_regress(train_data, f'{output_file_name}_{seed}',
                                                          model_fn=model_fn, test_data=test_data,
                                                          random_state=seed)
        else:
            train_y_truths, train_y_preds, test_y_truths, test_y_preds = multi_regress(train_data,
                                                                                       f'{output_file_name}_{seed}',
                                                                                       model_fn=model_fn,
                                                                                       test_data=test_data,
                                                                                       random_state=random_state)

            test_history['dpearsonr'].append(
                np.mean([pearsonr(test_y_truths[:, i][~ma.masked_invalid(test_y_truths[:, i]).mask],
                                  test_y_preds[:, i][~ma.masked_invalid(test_y_truths[:, i]).mask])[
                             0]
                         for i in range(test_y_truths.shape[1])]).item())
            test_history['cpearsonr'].append(
                np.mean([pearsonr(test_y_truths[i, :][~ma.masked_invalid(test_y_truths[i, :]).mask],
                                  test_y_preds[i, :][~ma.masked_invalid(test_y_truths[i, :]).mask])[
                             0]
                         for i in range(test_y_truths.shape[0])]).item())
            test_history['dspearman'].append(
                np.mean([spearmanr(test_y_truths[:, i][~ma.masked_invalid(test_y_truths[:, i]).mask],
                                   test_y_preds[:, i][~ma.masked_invalid(test_y_truths[:, i]).mask])[
                             0]
                         for i in range(test_y_truths.shape[1])]).item())
            test_history['cspearman'].append(
                np.mean([spearmanr(test_y_truths[i, :][~ma.masked_invalid(test_y_truths[i, :]).mask],
                                   test_y_preds[i, :][~ma.masked_invalid(test_y_truths[i, :]).mask])[
                             0]
                         for i in range(test_y_truths.shape[0])]).item())
            test_history['drmse'].append(
                np.mean(np.nanmean(np.square((test_y_truths - test_y_preds)), axis=0)).item())
            test_history['crmse'].append(
                np.mean(np.nanmean(np.square((test_y_truths - test_y_preds)), axis=1)).item())

        train_history['dpearsonr'].append(
            np.mean([pearsonr(train_y_truths[:, i][~ma.masked_invalid(train_y_truths[:, i]).mask],
                              train_y_preds[:, i][~ma.masked_invalid(train_y_truths[:, i]).mask])[
                         0]
                     for i in range(train_y_truths.shape[1])]).item())
        train_history['cpearsonr'].append(
            np.mean([pearsonr(train_y_truths[i, :][~ma.masked_invalid(train_y_truths[i, :]).mask],
                              train_y_preds[i, :][~ma.masked_invalid(train_y_truths[i, :]).mask])[
                         0]
                     for i in range(train_y_truths.shape[0])]).item())
        train_history['dspearman'].append(
            np.mean([spearmanr(train_y_truths[:, i][~ma.masked_invalid(train_y_truths[:, i]).mask],
                               train_y_preds[:, i][~ma.masked_invalid(train_y_truths[:, i]).mask])[
                         0]
                     for i in range(train_y_truths.shape[1])]).item())
        train_history['cspearman'].append(
            np.mean([spearmanr(train_y_truths[i, :][~ma.masked_invalid(train_y_truths[i, :]).mask],
                               train_y_preds[i, :][~ma.masked_invalid(train_y_truths[i, :]).mask])[
                         0]
                     for i in range(train_y_truths.shape[0])]).item())
        train_history['drmse'].append(np.mean(np.nanmean(np.square((train_y_truths - train_y_preds)), axis=0)).item())
        train_history['crmse'].append(np.mean(np.nanmean(np.square((train_y_truths - train_y_preds)), axis=1)).item())

    return (train_history, test_history)


if __name__ == '__main__':
    data_provider = DataProvider()
    labeled_samples, mut_only_labeled_samples = data_provider.get_labeled_samples()
    labeled_target_df = data_provider.target_df.loc[labeled_samples]
    labeled_mut_only_target_df = data_provider.target_df.loc[mut_only_labeled_samples]
    label_gex_df = data_provider.gex_dat.loc[labeled_samples]
    label_mut_df = data_provider.mut_dat.loc[labeled_samples]
    label_mut_only_df = data_provider.mut_dat.loc[mut_only_labeled_samples]

    train_gex_data = (label_gex_df, labeled_target_df)
    train_mut_data = (label_mut_df, labeled_target_df)
    test_data = (label_mut_only_df, labeled_mut_only_target_df)

    gex_train_history, _ = n_time_cv_regress(train_gex_data, 'gex_pred', n=5, model_fn=regress_with_enet,
                                             test_data=None, random_state=2020)

    with open('./predictions/gex_pred.json', 'w') as f:
        json.dump(gex_train_history, f)

    mut_train_history, mut_test_history = n_time_cv_regress(train_mut_data, 'mut_pred', n=5, model_fn=regress_with_enet,
                                                            test_data=test_data, random_state=2020)

    with open('./predictions/mut_pred.json', 'w') as f:
        json.dump(mut_train_history, f)

    with open('./predictions/test_mut_pred.json', 'w') as f:
        json.dump(mut_test_history, f)
