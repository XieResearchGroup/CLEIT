import os
import gzip
import argparse
import numpy as np
import pandas as pd
import data_config
import preprocess_ccle_gdsc_utils
import preprocess_xena_utils


def align_feature(df1, df2):
    matched_features = list(set(df1.columns.tolist()) & set(df2.columns.tolist()))
    matched_features.sort()
    print('Aligned dataframes have {} features in common'.format(len(matched_features)))
    return df1[matched_features], df2[matched_features]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Multi-omics data preprocessing')
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--propagation', dest='propagation', action='store_true')
    parser.add_argument('--no-propagation', dest='propagation', action='store_false')
    parser.set_defaults(propagation=True)
    parser.add_argument('--target', dest='target', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--filter', dest='filter', nargs='?', default='ALL', choices=['MAD', 'ALL'])

    args = parser.parse_args()

    output_folder = os.path.join(data_config.preprocessed_data_folder, args.filter)
    # After imputation
    gdsc_target_dat = preprocess_ccle_gdsc_utils.preprocess_target_data()

    xena_gex_dat = preprocess_xena_utils.preprocess_gex_df(file_path=data_config.xena_gex_file)
    ccle_gex_dat = preprocess_ccle_gdsc_utils.preprocess_ccle_gex_df(file_path=data_config.ccle_gex_file)
    xena_gex_dat, ccle_gex_dat = align_feature(xena_gex_dat, ccle_gex_dat)

    xena_mut_dat = preprocess_xena_utils.preprocess_mut(propagation_flag=args.propagation, data_config.xena_mut_file)
    ccle_mut_dat = preprocess_ccle_gdsc_utils.preprocess_ccle_mut(propagation_flag=args.propagation,
                                                                  data_config.ccle_mut_file)
    xena_mut_dat, ccle_mut_dat = align_feature(xena_mut_dat, ccle_mut_dat)

    labeled_mut_file_name = 'labeled_mut'
    unlabeled_mut_file_name = 'unlabeled_mut'
    labeled_gex_file_name = 'labeled_gex'
    unlabeled_gex_file_name = 'unlabeled_gex'
    labeled_target_file_name = 'labeled_target'

    if args.propagation:
        labeled_mut_file_name = labeled_mut_file_name + '_propagated'
        unlabeled_mut_file_name = unlabeled_mut_file_name + '_propagated'
        labeled_gex_file_name = labeled_gex_file_name + '_propagated'
        unlabeled_gex_file_name = unlabeled_gex_file_name + '_propagated'

    # align features across domains
    # xena_gex_dat, xena_mut_dat = align_feature(xena_gex_dat, xena_mut_dat)
    # ccle_gex_dat, ccle_mut_dat = align_feature(ccle_gex_dat, ccle_mut_dat)
    # select top 5000 features according to MAD of xena gene expression data, and align features
    if args.filter == 'MAD':
        xena_gex_dat = preprocess_xena_utils.filter_with_MAD(xena_gex_dat)
    xena_gex_dat, ccle_gex_dat = align_feature(xena_gex_dat, ccle_gex_dat)
    # xena_gex_dat, xena_mut_dat = align_feature(xena_gex_dat, xena_mut_dat)
    # ccle_gex_dat, ccle_mut_dat = align_feature(ccle_gex_dat, ccle_mut_dat)
    # indentify unlabeled and labeled data respectively
    ccle_target_samples = list(
        set(ccle_gex_dat.index.to_list()) & set(ccle_mut_dat.index.to_list()) & set(gdsc_target_dat.index.to_list()))
    # labeled data and output
    labeled_gex_dat = ccle_gex_dat.loc[ccle_target_samples, :]
    labeled_mut_dat = ccle_mut_dat.loc[ccle_target_samples, :]
    labeled_targets_dat = gdsc_target_dat.loc[ccle_target_samples, :]
    labeled_gex_dat.to_csv(os.path.join(output_folder, labeled_gex_file_name + '.csv'), index_label='Sample')
    labeled_mut_dat.to_csv(os.path.join(output_folder, labeled_mut_file_name + '.csv'), index_label='Sample')
    labeled_targets_dat.to_csv(os.path.join(output_folder, labeled_target_file_name + '_' + args.target + '.csv'),
                               index_label='Sample')

    # unlabeled data and output
    unlabeled_gex_dat = pd.concat([xena_gex_dat, ccle_gex_dat.loc[~ccle_gex_dat.index.isin(ccle_target_samples), :]])
    unlabeled_mut_dat = pd.concat([xena_mut_dat, ccle_mut_dat.loc[~ccle_mut_dat.index.isin(ccle_target_samples), :]])
    unlabeled_gex_dat.to_csv(os.path.join(output_folder, unlabeled_gex_file_name + '.csv'), index_label='Sample')
    unlabeled_mut_dat.to_csv(os.path.join(output_folder, unlabeled_mut_file_name + '.csv'), index_label='Sample')
