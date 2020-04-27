from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# import logging
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import argparse
import os

import data
import data_config
import preprocess_ccle_gdsc_utils
import preprocess_xena_utils


def process_with_rf(train_features, y_train, cv_split_rf):
    try:
        # logger.debug("Training Random Forest model")
        # mx_depth: trees' maximum depth
        # n_estimators: number of trees to use
        # n_jobs = -1 means to run the jobs in parallel
        rf_tuning_parameters = [{'n_estimators': [10, 100, 500], 'max_depth': [10, 50, 100]}]
        #rf_tuning_parameters = [{'n_estimators': [5], 'max_depth': [10]}]
        rf = GridSearchCV(RandomForestRegressor(min_samples_split=10), rf_tuning_parameters, n_jobs=-1, cv=cv_split_rf,
                          verbose=2)
        rf.fit(train_features, y_train)  # , groups=train_groups
        # logger.debug("Trained Random Forest successfully")
        return rf

    except Exception as e:
        # logger.debug("Fail to Train Random Forest, caused by %s" % e.message)
        raise e


def process_with_enet(train_features, y_train, cv_split_enet):
    try:
        # logger.debug("Training elastic net regression model")
        alphas = np.logspace(-5, -1, num=3, endpoint=True)
        l1_ratios = np.array([0.1, 1, 10])
        enet = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=cv_split_enet, n_jobs=-1, verbose=2,
                            normalize=True,
                            max_iter=10000)
        enet.fit(train_features, y_train)
        # logger.debug("Trained Elastic net regression model successfully")
        return enet
    except Exception as e:
        raise e


def process_with_xgb(train_features, y_train, cv_split_xgb):
    try:
        xgb_tuning_parameters = {
            'learning_rate': [0.01, 0.5, 0.9],
            'n_estimators': {50, 100, 500},
            'subsample': [0.3, 0.5, 0.9]
        }
        gbm = GridSearchCV(estimator=xgb.XGBRegressor(),
                           param_grid=xgb_tuning_parameters,
                           scoring='neg_mean_squared_error',
                           cv=cv_split_xgb,
                           verbose=2)
        gbm.fit(train_features, y_train)
        return gbm
    except Exception as e:
        raise e


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Machine Learning baseline approaches')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--propagation', dest='propagation', action='store_true')
    parser.add_argument('--no-propagation', dest='propagation', action='store_false')
    parser.set_defaults(propagation=True)
    parser.add_argument('--method', dest='method', nargs='?', default='rf', choices=['rf', 'enet', 'xgb'])
    parser.add_argument('--target', dest='target', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--filter', dest='filter', nargs='?', default='FILE', choices=['MAD', 'FILE'])
    parser.add_argument('--feat_num', dest='feature_number', nargs='?', default=5000)

    args = parser.parse_args()
    # if args.filter == 'MAD':
    #     feature_filter_fn = preprocess_xena_utils.filter_with_MAD
    # else:
    #     feature_filter_fn = None

    if args.method == 'enet':
        model_fn = process_with_enet
    elif args.method == 'xgb':
        model_fn = process_with_xgb
    else:
        model_fn = process_with_rf

    data_provider = data.DataProvider(feature_filter=args.filter, target=args.target,
                                      feature_number=args.feature_number,
                                      omics=['gex', 'mut'], scale_fn=data.min_max_scale)
    data_provider.labeled_data['gex'].columns = data_provider.labeled_data['gex'].columns + '_gex'
    data_provider.labeled_data['mut'].columns = data_provider.labeled_data['mut'].columns + '_mut'

    mut_test_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_test_data['target'], fill_value=-1),
                                     index=data_provider.labeled_test_data['target'].index,
                                     columns=data_provider.labeled_test_data['target'].columns)

    mut_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_data['target'], fill_value=-1),
                                     index=data_provider.labeled_data['target'].index,
                                     columns=data_provider.labeled_data['target'].columns)

    gex_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_data['target'], fill_value=-1),
                                     index=data_provider.labeled_data['target'].index,
                                     columns=data_provider.labeled_data['target'].columns)

    overlapped_prediction_df = pd.DataFrame(np.full_like(data_provider.labeled_data['target'], fill_value=-1),
                                            index=data_provider.labeled_data['target'].index,
                                            columns=data_provider.labeled_data['target'].columns)

    for drug in data_provider.labeled_data['target'].columns:
        print("Training: {}".format(drug))
        y = data_provider.labeled_data['target'].loc[~data_provider.labeled_data['target'][drug].isna(), drug]
        sample_ids = data_provider.labeled_data['mut'].index.intersection(y.index)
        X_mut = data_provider.labeled_data['mut'].loc[sample_ids,]
        X_gex = data_provider.labeled_data['gex'].loc[sample_ids,]
        X = pd.concat([X_mut, X_gex], axis=1, join='inner')
        # print(X.columns)
        y = y.reindex(sample_ids)

        outer_kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
        cv_split = outer_kfold.split(X_mut)
        try:
            print('Mutation-Only Training')
            trained_model = model_fn(X_mut, y, list(cv_split))
            prediction = trained_model.predict(data_provider.labeled_test_data['mut'])
            mut_test_prediction_df.loc[data_provider.labeled_test_data['mut'].index, drug] = prediction
        except Exception as e:
            print(e)

        outer_kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
        for train_index, test_index in outer_kfold.split(y):
            X_mut_train, X_mut_test = X_mut.iloc[train_index, :], X_mut.iloc[test_index, :]
            X_gex_train, X_gex_test = X_gex.iloc[train_index, :], X_gex.iloc[test_index, :]
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]

            y_train, y_test = y[train_index], y[test_index]

            #kfold = KFold(n_splits=3, shuffle=True, random_state=2020)
            #cv_split = kfold.split(X)
            #try:
            #    print('Overlapped Training')
            #    trained_model = model_fn(X_train, y, list(cv_split))
            #    prediction = trained_model.predict(X_test)
            #    overlapped_prediction_df.loc[y.index[test_index], drug] = prediction
            #except Exception as e:
            #    print(e)

            kfold = KFold(n_splits=3, shuffle=True, random_state=2020)
            cv_split = kfold.split(X_gex)
            try:
                print('Gex Training')
                trained_model = model_fn(X_gex_train, y, list(cv_split))
                prediction = trained_model.predict(X_gex_test)
                gex_prediction_df.loc[y.index[test_index], drug] = prediction
            except Exception as e:
                print(e)

            kfold = KFold(n_splits=3, shuffle=True, random_state=2020)
            cv_split = kfold.split(X_mut)
            try:
                print('Mutation Training')
                trained_model = model_fn(X_mut_train, y, list(cv_split))
                prediction = trained_model.predict(X_mut_test)
                mut_prediction_df.loc[y.index[test_index], drug] = prediction
            except Exception as e:
                print(e)

    mut_test_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.method + '_mut_only' + str(
        args.propagation) + '_' + args.filter + '_prediction.csv'),
                                    index_label='Sample')

    overlapped_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.method + '_both_' + str(
        args.propagation) + '_' + args.filter + '_multi_regression_5fold_prediction.csv'),
                                    index_label='Sample')

    gex_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.method + '_gex_' + str(
        args.propagation) + '_' + args.filter + '_multi_regression_5fold_prediction.csv'),
                                    index_label='Sample')

    mut_prediction_df.to_csv(os.path.join('predictions', args.target + '_' + args.method + '_mut_' + str(
        args.propagation) + '_' + args.filter + '_multi_regression_5fold_prediction.csv'),
                                    index_label='Sample')
