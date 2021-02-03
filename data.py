import os
import datetime
import numpy as np
import random
import pandas as pd
import data_config
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold

_RNG_SEED = None

DRUG_DICT = {
    'gem': 'gemcitabine',
    'ava': 'avagacestat',
}


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


class DataProvider:
    def __init__(self, batch_size=64, target='AUC', random_seed=2019):
        self.seed = random_seed
        self.target = target
        self.batch_size = batch_size
        self._load_gex_data()
        self._load_mut_data()
        self._load_target_data()
        self.shape_dict = {'gex': self.gex_dat.shape[-1],
                           'mut': self.mut_dat.shape[-1],
                           'target': self.target_df.shape[-1]}

    def _load_gex_data(self):
        self.gex_dat = pd.read_csv(data_config.gex_feature_file, index_col=0)
        # ccle_sample_info_df = pd.read_csv(data_config.ccle_sample_file, index_col=0)
        # with gzip.open(data_config.xena_sample_file) as f:
        #     xena_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
        # xena_samples = xena_sample_info_df.index.intersection(self.gex_dat.index)
        # ccle_samples = self.gex_dat.index.difference(xena_samples)
        # xena_sample_info_df = xena_sample_info_df.loc[xena_samples]
        # ccle_sample_info_df = ccle_sample_info_df.loc[ccle_samples.intersection(ccle_sample_info_df.index)]
        # self.xena_gex_df = self.gex_dat.loc[xena_samples]
        # self.mut_gex_df = self.gex_dat.loc[ccle_samples]

    def _load_mut_data(self):
        self.xena_mut_dat = pd.read_csv(data_config.xena_mut_uq_file, index_col=0)
        self.ccle_mut_dat = pd.read_csv(data_config.ccle_mut_uq_file, index_col=0)
        self.mut_dat = self.xena_mut_dat.append(self.ccle_mut_dat)

    def _load_target_data(self):
        # gdsc1_response = pd.read_csv(data_config.gdsc_target_file1)
        # gdsc2_response = pd.read_csv(data_config.gdsc_target_file2)
        # gdsc1_sensitivity_df = gdsc1_response[['COSMIC_ID', 'DRUG_NAME', self.target]]
        # gdsc2_sensitivity_df = gdsc2_response[['COSMIC_ID', 'DRUG_NAME', self.target]]
        # gdsc1_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc1_sensitivity_df['DRUG_NAME'].str.lower()
        # gdsc2_sensitivity_df.loc[:, 'DRUG_NAME'] = gdsc2_sensitivity_df['DRUG_NAME'].str.lower()
        #
        # if self.target == 'LN_IC50':
        #     gdsc1_sensitivity_df.loc[:, self.target] = np.exp(gdsc1_sensitivity_df[self.target])
        #     gdsc2_sensitivity_df.loc[:, self.target] = np.exp(gdsc2_sensitivity_df[self.target])
        #
        # gdsc1_target_df = gdsc1_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
        # gdsc2_target_df = gdsc2_sensitivity_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
        # gdsc1_target_df = gdsc1_target_df.loc[gdsc1_target_df.index.difference(gdsc2_target_df.index)]
        # gdsc_target_df = pd.concat([gdsc1_target_df, gdsc2_target_df])
        target = self.target.lower()
        gdsc_target_df = pd.read_csv(data_config.gdsc_target_file)
        gdsc_target_df = gdsc_target_df[['COSMIC_ID', 'DRUG_NAME', target]]
        gdsc_target_df.dropna(subset=[target], inplace=True)
        gdsc_target_df = gdsc_target_df.groupby(['COSMIC_ID', 'DRUG_NAME']).mean()
        target_df = gdsc_target_df.reset_index().pivot_table(values=target, index='COSMIC_ID', columns='DRUG_NAME')

        ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
        ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
        ccle_sample_info.index = ccle_sample_info.index.astype('int')

        gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
        gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
        gdsc_sample_info.index = gdsc_sample_info.index.astype('int')

        gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
            ['DepMap_ID']]
        gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

        target_df.index = target_df.index.map(gdsc_sample_mapping_dict)
        target_df = target_df.loc[target_df.index.dropna()]
        gex_labeled_samples = self.gex_dat.index.intersection(target_df.index)

        target_df.drop(columns=target_df.columns[
            target_df.loc[gex_labeled_samples].isna().sum() / len(gex_labeled_samples) >= 0.1], inplace=True)
        self.target_df = target_df

    def get_unlabeled_gex_dataloader(self):
        gex_dataset = TensorDataset(torch.from_numpy(self.gex_dat.values.astype('float32')))
        unlabeled_gex_dataloader = DataLoader(gex_dataset,
                                              batch_size=self.batch_size,
                                              shuffle=True)

        return unlabeled_gex_dataloader

    def get_labeled_gex_dataloader(self):
        gex_labeled_samples = self.gex_dat.index.intersection(self.target_df.index)
        gex_target_df = self.target_df.loc[gex_labeled_samples]
        gex_labeled_samples = gex_labeled_samples[gex_target_df.shape[1] - gex_target_df.isna().sum(axis=1) >= 2]
        gex_target_df = self.target_df.loc[gex_labeled_samples]

        labeled_gex_dataset = TensorDataset(
            torch.from_numpy(self.gex_dat.loc[gex_labeled_samples].values.astype('float32')),
            torch.from_numpy(gex_target_df.values.astype('float32'))
        )
        labeled_gex_dataloader = DataLoader(labeled_gex_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        return labeled_gex_dataloader

    def get_drug_labeled_gex_dataloader(self, drug=None, ft_flag=True):
        # drug = DRUG_DICT[drug]
        # drug_target_df = self.target_df[drug]
        # drug_target_df.dropna(inplace=True)
        # drug_gex_labeled_samples = self.gex_dat.index.intersection(drug_target_df.index)
        # # get gex dataset and dataloader
        # drug_gex_target_df = drug_target_df.loc[drug_gex_labeled_samples]
        # gex_label_vec = (drug_gex_target_df < np.median(drug_gex_target_df)).astype('int')
        gex_labeled_samples = self.gex_dat.index.intersection(self.target_df.index)
        gex_target_df = self.target_df.loc[gex_labeled_samples]
        gex_labeled_samples = gex_labeled_samples[gex_target_df.shape[1] - gex_target_df.isna().sum(axis=1) >= 2]
        gex_target_df = self.target_df.loc[gex_labeled_samples]

        sample_label_vec = (gex_target_df.isna().sum(axis=1) <= gex_target_df.isna().sum(axis=1).median()).astype('int')

        if not ft_flag:
            pass

        else:
            s_kfold = StratifiedKFold(n_splits=5, random_state=self.seed)
            for train_index, test_index in s_kfold.split(self.gex_dat.loc[gex_labeled_samples].values,
                                                         sample_label_vec):
                train_labeled_df, test_labeled_df = self.gex_dat.loc[gex_labeled_samples].values[train_index], \
                                                    self.gex_dat.loc[gex_labeled_samples].values[test_index]
                train_labels, test_labels = gex_target_df.values[train_index].astype('float32'), gex_target_df.values[
                    test_index].astype('float32')

                train_labeled_dateset = TensorDataset(
                    torch.from_numpy(train_labeled_df.astype('float32')),
                    torch.from_numpy(train_labels))
                test_labeled_dateset = TensorDataset(
                    torch.from_numpy(test_labeled_df.astype('float32')),
                    torch.from_numpy(test_labels))

                train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)

                test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True)

                yield train_labeled_dataloader, test_labeled_dataloader

    def get_drug_labeled_mut_dataloader(self, drug=None, ft_flag=True):
        # drug = DRUG_DICT[drug]
        # drug_target_df = self.target_df[drug]
        # drug_target_df.dropna(inplace=True)
        # drug_gex_labeled_samples = self.gex_dat.index.intersection(drug_target_df.index)
        # drug_mut_labeled_samples = self.ccle_mut_dat.index.intersection(drug_target_df.index)
        # drug_mut_only_labeled_samples = drug_mut_labeled_samples.difference(drug_gex_labeled_samples)
        # drug_mut_labeled_samples = drug_mut_labeled_samples.difference(drug_mut_only_labeled_samples)
        #
        # drug_mut_target_df = drug_target_df.loc[drug_mut_labeled_samples]
        # mut_label_vec = (drug_mut_target_df < np.median(drug_mut_target_df)).astype('int')
        gex_labeled_samples = self.gex_dat.index.intersection(self.target_df.index)
        mut_labeled_samples = self.ccle_mut_dat.index.intersection(self.target_df.index)
        mut_only_labeled_samples = mut_labeled_samples.difference(gex_labeled_samples)
        mut_labeled_samples = mut_labeled_samples.difference(mut_only_labeled_samples)

        mut_target_df = self.target_df.loc[mut_labeled_samples]
        mut_labeled_samples = mut_labeled_samples[mut_target_df.shape[1] - mut_target_df.isna().sum(axis=1) >= 2]
        mut_target_df = self.target_df.loc[mut_labeled_samples]

        mut_only_target_df = self.target_df.loc[mut_only_labeled_samples]
        mut_only_labeled_samples = mut_only_labeled_samples[
            mut_only_target_df.shape[1] - mut_only_target_df.isna().sum(axis=1) >= 2]
        mut_only_target_df = self.target_df.loc[mut_only_labeled_samples]

        sample_label_vec = (mut_target_df.isna().sum(axis=1) <= mut_target_df.isna().sum(axis=1).median()).astype('int')

        labeled_drug_mut_only_dataset = TensorDataset(
            torch.from_numpy(self.ccle_mut_dat.loc[mut_only_labeled_samples].values.astype('float32')),
            torch.from_numpy(mut_only_target_df.values.astype('float32'))
        )

        labeled_drug_mut_only_dataloader = DataLoader(labeled_drug_mut_only_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)
        if not ft_flag:
            pass
            # labeled_mut_dataset = TensorDataset(
            #     torch.from_numpy(self.ccle_mut_dat.loc[drug_mut_labeled_samples].values.astype('float32')),
            #     torch.from_numpy(drug_mut_target_df.values.astype('float32'))
            # )
            # labeled_mut_dataloader = DataLoader(labeled_mut_dataset,
            #                                     batch_size=self.batch_size,
            #                                     shuffle=True)
            # return labeled_mut_dataloader, labeled_drug_mut_only_dataloader

        else:
            s_kfold = StratifiedKFold(n_splits=5, random_state=self.seed)
            for train_index, test_index in s_kfold.split(self.ccle_mut_dat.loc[mut_labeled_samples].values,
                                                         sample_label_vec):
                train_labeled_df, test_labeled_df = self.ccle_mut_dat.loc[mut_labeled_samples].values[train_index], \
                                                    self.ccle_mut_dat.loc[mut_labeled_samples].values[test_index]
                train_labels, test_labels = mut_target_df.values[train_index].astype('float32'), mut_target_df.values[
                    test_index].astype('float32')

                train_labeled_dateset = TensorDataset(
                    torch.from_numpy(train_labeled_df.astype('float32')),
                    torch.from_numpy(train_labels))
                test_labeled_dateset = TensorDataset(
                    torch.from_numpy(test_labeled_df.astype('float32')),
                    torch.from_numpy(test_labels))

                train_labeled_dataloader = DataLoader(train_labeled_dateset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True)

                test_labeled_dataloader = DataLoader(test_labeled_dateset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True)

                yield train_labeled_dataloader, test_labeled_dataloader, labeled_drug_mut_only_dataloader

    def get_unlabeld_mut_dataloader(self, match=True):
        if match:
            mut_gex_samples = self.gex_dat.index.intersection(self.mut_dat.index)
            mut_gex_dataset = TensorDataset(
                torch.from_numpy(self.mut_dat.loc[mut_gex_samples].values.astype('float32')),
                torch.from_numpy(self.gex_dat.loc[mut_gex_samples].values.astype('float32'))
            )
            unlabeled_mut_gex_dataloader = DataLoader(mut_gex_dataset,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      drop_last=True
                                                      )
            return unlabeled_mut_gex_dataloader

        else:
            mut_dataset = TensorDataset(torch.from_numpy(self.mut_dat.values.astype('float32')))
            unlabeled_mut_dataloader = DataLoader(mut_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

            return unlabeled_mut_dataloader
