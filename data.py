import os
import datetime

import numpy as np
import random
import pandas as pd
import data_config
import preprocess_ccle_gdsc_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

_RNG_SEED = None


def get_rng(obj=None):
    """
    This function is copied from `tensorpack
    <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
    Get a good RNG seeded with time, pid and the object.
    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return random.Random(seed)


def min_max_scale(X_train):
    preprocessor = MinMaxScaler((0, 1)).fit(X_train)
    # X_train = preprocessor.transform(X_train)
    # X_test = preprocessor.transform(X_test)
    return preprocessor


def standardize_scale(X_train):
    preprocessor = StandardScaler().fit(X_train)
    # X_train = preprocessor.transform(X_train)
    # X_test = preprocessor.transform(X_test)
    return preprocessor


def sample_min_max_scale(df):
    min_vec = df.min(axis=1)
    max_vec = df.max(axis=1)
    return df.sub(min_vec, axis=0).div(max_vec - min_vec, axis=0)


def sample_standardize_scale(df):
    mean_vec = df.mean(axis=1)
    std_vec = df.std(axis=1)
    return df.sub(mean_vec, axis=0).div(std_vec, axis=0)


def align_feature(df1, df2):
    matched_features = list(set(df1.columns.tolist()) & set(df2.columns.tolist()))
    matched_features.sort()
    print('Aligned dataframes have {} features in common'.format(len(matched_features)))
    return df1[matched_features], df2[matched_features]


class DataProvider:
    def __init__(self, feature_filter_fn=None, feature_number=5000, propagation=True, target='AUC', scale_fn=None,
                 omics=None,
                 random_seed=2019):
        self.omics = omics
        self.random_seed = random_seed
        self.feature_filter_fn = feature_filter_fn
        self.feature_number = feature_number
        self.propagation = propagation
        self.target = target
        self.scale_fn = scale_fn
        self.labeled_data, self.unlabeled_data = self._load_data()
        self.shape_dict = self._get_shape_dict()
        self.matched_index = self._get_matched_index()

    def _load_data(self):
        xena_gex_dat = pd.read_csv(data_config.xena_preprocessed_gex_file + '.csv', index_col=0)
        gex_dat = preprocess_ccle_gdsc_utils.preprocess_ccle_gex_df(file_path=data_config.ccle_gex_file)
        xena_gex_dat, gex_dat = align_feature(xena_gex_dat, gex_dat)
        if self.propagation:
            xena_mut_dat = pd.read_csv(data_config.xena_preprocessed_mut_file + '_propagated.csv', index_col=0)
        else:
            xena_mut_dat = pd.read_csv(data_config.xena_preprocessed_mut_file + '.csv', index_col=0)

        mut_dat = preprocess_ccle_gdsc_utils.preprocess_ccle_mut(propagation_flag=self.propagation,
                                                                 mutation_dat_file=data_config.ccle_mut_file)
        xena_mut_dat, mut_dat = align_feature(xena_mut_dat, mut_dat)
        if self.feature_filter_fn:
            xena_gex_dat = self.feature_filter_fn(xena_gex_dat, k=self.feature_number)
            xena_gex_dat, gex_dat = align_feature(xena_gex_dat, gex_dat)
            xena_mut_dat = self.feature_filter_fn(xena_mut_dat, k=self.feature_number)
            xena_mut_dat, mut_dat = align_feature(xena_mut_dat, mut_dat)

        target_dat = preprocess_ccle_gdsc_utils.preprocess_target_data(score=self.target)
        target_samples = list(
            set(gex_dat.index.to_list()) & set(mut_dat.index.to_list()) & set(target_dat.index.to_list()))

        labeled_gex_dat = gex_dat.loc[target_samples, :]
        labeled_mut_dat = mut_dat.loc[target_samples, :]
        labeled_targets_dat = target_dat.loc[target_samples, :]
        unlabeled_gex_dat = pd.concat(
            [xena_gex_dat, gex_dat.loc[~gex_dat.index.isin(target_samples), :]])
        unlabeled_mut_dat = pd.concat(
            [xena_mut_dat, mut_dat.loc[~mut_dat.index.isin(target_samples), :]])

        if self.scale_fn:
            self.gex_scaler = min_max_scale(xena_gex_dat)
            self.mut_scaler = min_max_scale(xena_mut_dat)

            labeled_gex_dat = pd.DataFrame(self.gex_scaler.transform(labeled_gex_dat),
                                           index=labeled_gex_dat.index, columns=labeled_gex_dat.columns)
            unlabeled_gex_dat = pd.DataFrame(self.gex_scaler.transform(unlabeled_gex_dat),
                                             index=unlabeled_gex_dat.index, columns=unlabeled_gex_dat.columns)
            labeled_mut_dat = pd.DataFrame(self.mut_scaler.transform(labeled_mut_dat),
                                           index=labeled_mut_dat.index, columns=labeled_mut_dat.columns)
            unlabeled_mut_dat = pd.DataFrame(self.mut_scaler.transform(unlabeled_mut_dat),
                                             index=unlabeled_mut_dat.index, columns=unlabeled_mut_dat.columns)

        labeled_data = {}
        unlabeled_data = {}
        labeled_data['mut'] = labeled_mut_dat
        labeled_data['gex'] = labeled_gex_dat
        labeled_data['target'] = labeled_targets_dat
        unlabeled_data['mut'] = unlabeled_mut_dat
        unlabeled_data['gex'] = unlabeled_gex_dat

        self.target_scaler = dict()
        for drug in labeled_data['target'].columns:
            self.target_scaler[drug] = min_max_scale(
                labeled_data['target'].loc[~labeled_data['target'][drug].isna(), drug].values.reshape(-1, 1))
            labeled_data['target'].loc[~labeled_data['target'][drug].isna(), drug] = np.squeeze(
                self.target_scaler[drug].transform(
                    labeled_data['target'].loc[~labeled_data['target'][drug].isna(), drug].values.reshape(-1, 1)))

        return labeled_data, unlabeled_data

    def _get_shape_dict(self):
        shape_dict = dict()
        for omic in self.omics:
            shape_dict[omic] = self.labeled_data[omic].shape[-1]
        shape_dict['target'] = self.labeled_data['target'].shape[-1]
        return shape_dict

    def _get_matched_index(self):
        matched_index = self.unlabeled_data[self.omics[0]].index
        for omic in self.omics[1:]:
            matched_index = matched_index.intersection(self.unlabeled_data[omic].index)
        return matched_index

    def get_k_folds(self, k=5):
        kfold = KFold(n_splits=k, shuffle=True, random_state=self.random_seed)
        cv_splits = list(kfold.split(self.labeled_data['target']))
        return cv_splits


