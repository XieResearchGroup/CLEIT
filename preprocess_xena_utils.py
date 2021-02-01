import os
import gzip
import numpy as np
import pandas as pd
import data_config

def filter_features(df, mean_tres=1.0, std_thres=0.5, cor_thres=None):
    """
    filter genes of low information burden
    first need to reverse transformation applied log2(x+0.001)
    mean: sparsity threshold, std: variability threshold
    :param df: samples X features
    :param mean_tres:
    :param std_thres:
    :param cor_thres:
    :return:
    """
    df = df.loc[:, df.apply(lambda col: col.isna().sum()) == 0]
    feature_stds = df.std()
    feature_means = df.mean()
    std_to_drop = feature_stds[list(np.where(feature_stds <= std_thres)[0])].index.tolist()
    mean_to_drop = feature_means[list(np.where(feature_means <= mean_tres)[0])].index.tolist()
    to_drop = list(set(std_to_drop) | set(mean_to_drop))
    df.drop(labels=to_drop, axis=1, inplace=True)
    return df

def filter_with_MAD(df, k=5000):
    result = df[(df - df.median()).abs().median().nlargest(k).index.tolist()]
    return result

def get_normal_samples(sample_info_file_path=data_config.xena_sample_file):
    with gzip.open(sample_info_file_path) as f:
        sample_df = pd.read_csv(f, sep='\t', index_col=0)
    sample_df.dropna(subset=['sample_type'], inplace=True)
    normal_samples_list = sample_df[sample_df['sample_type'].apply(lambda x: "Normal" in x)].index.to_list()
    if len(normal_samples_list) > 0:
        return normal_samples_list
    else:
        return None

def preprocess_gex_df(file_path=data_config.xena_gex_file, df=None, MAD=False, feature_num = 5000, feature_list=None,
                      output_file_path=None, mapping_file=data_config.xena_id_mapping_file):
    if df is None:
        # mapping_df = get_id_mapping()
        with gzip.open(filename=file_path) as f:
            df = pd.read_csv(f, sep='\t', index_col=0)
        normal_samples = get_normal_samples(data_config.xena_sample_file)
        if normal_samples:
            df.drop(columns=df.columns.intersection(normal_samples), inplace=True)
        #df.index = df.index.map(lambda s: s[:s.find('.')])
        df = 2 ** df - 0.001
        if mapping_file:
            print('Start Mapping')
            mapping_df = pd.read_csv(mapping_file, sep='\t',index_col=0)
            mapping_dict = mapping_df['gene'].to_dict()
            df.index = df.index.map(mapping_dict)
            if any(df.index.isna()):
                print('removing NA')
                df = df.loc[df.index.dropna()]
            print('start grouping')
            df = df.groupby(level=0).mean()
        # df = df.groupby('gene').mean()
        df = df.transpose()
        df = np.log1p(df)
        df = np.clip(df, a_min=0, a_max=None)
        df.index.name = 'Sample'
    if MAD:
        df = df.transpose()
        df = np.exp(df)
        df = filter_features(df)
        df = np.log1p(df-1.0)
        df = df.transpose()
        df = filter_with_MAD(df, k=feature_num)
        if output_file_path:
            output_file_path = output_file_path + '_MAD'
    if feature_list is not None:
        genes_to_keep = list(set(df.columns.tolist()) & set(feature_list))
        df = df[genes_to_keep]
        if output_file_path:
            output_file_path = output_file_path + '_filtered'
    df = df.astype('float32')
    print('Preprocessed data has {0} samples and {1} features'.format(df.shape[0], df.shape[1]))
    if output_file_path:
        df.to_csv(output_file_path + '.csv', index_label='Sample')
    return df

def load_xena_mutation_df(file_path, network_id_file=None):
    with gzip.open(file_path) as f:
        mutation_df = pd.read_csv(f, sep='\t', index_col=0)
    mutation_df = mutation_df.transpose()
    if network_id_file:
        with gzip.open(network_id_file) as f:
            mapping_df = pd.read_csv(f, sep='\t', index_col=0)
        mutation_df = mutation_df[mutation_df.columns.intersection(mapping_df.preferred_name)]
    print('Preprocessed data has {0} samples and {1} features'.format(mutation_df.shape[0], mutation_df.shape[1]))
    return mutation_df

def preprocess_mut(propagation_flag=True, mutation_dat_file=data_config.xena_mut_file,
                   kernel_file=data_config.propagation_kernel_file, network_id_file=data_config.string_id_mapping_file,
                   output_file_path=None):
    if propagation_flag:
        binary_mutation_df = load_xena_mutation_df(mutation_dat_file, network_id_file=network_id_file)
        print('Propagation Start!')

        kernel = pd.read_feather(kernel_file, columns=binary_mutation_df.columns.union(['index']))
        kernel.set_index('index', inplace=True)
        kernel.index = [ind.decode() for ind in kernel.index]
        kernel = kernel.transpose()

        propagation_result = pd.DataFrame(np.zeros((binary_mutation_df.shape[0], kernel.shape[1])),
                                          index=binary_mutation_df.index, columns=kernel.columns)
        to_drop = []
        for sample_id in propagation_result.index:
            features_to_propagate = binary_mutation_df.columns[binary_mutation_df.loc[sample_id] == 1]
            # print(features_to_propagate)
            features_to_propagate = kernel.index.intersection(features_to_propagate)
            # print(features_to_propagate)
            if len(features_to_propagate) > 0:
                propagation_result.loc[sample_id] = kernel.loc[features_to_propagate].sum(axis=0)
            else:
                to_drop.append(sample_id)
        if len(to_drop) > 0:
            propagation_result.drop(to_drop, inplace=True)
        propagation_result = propagation_result.astype('float32')
        if output_file_path:
            print('Preprocessed data has {0} samples and {1} features'.format(propagation_result.shape[0], propagation_result.shape[1]))
            propagation_result.to_csv(output_file_path+'_propagated.csv', index_label='Sample')
        return propagation_result
    else:
        binary_mutation_df = load_xena_mutation_df(mutation_dat_file)
        if output_file_path:
            print('Preprocessed data has {0} samples and {1} features'.format(binary_mutation_df.shape[0], binary_mutation_df.shape[1]))
            binary_mutation_df.to_csv(output_file_path + '.csv', index_label='Sample')
        return binary_mutation_df

if __name__=='__main__':
    preprocess_gex_df(output_file_path=data_config.xena_preprocessed_gex_file)
    #preprocess_mut(output_file_path=data_config.xena_preprocessed_mut_file)