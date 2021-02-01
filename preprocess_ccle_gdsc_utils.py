import os
import gzip
import numpy as np
import pandas as pd
import data_config
import string
import feather

# def normalize_celline_names(s):
#     s = s.upper()
#     translator = str.maketrans('', '', string.punctuation + ' ')
#     return s.translate(translator)
#
#
# def get_normalized_ccle_sample_dict(file_path=data_config.ccle_sample_file):
#     ccle_celline_info = pd.read_csv(file_path, sep='\t')
#     ccle_celline_info['Cell line primary name'] = ccle_celline_info['Cell line primary name'].map(
#         normalize_celline_names)
#     ccle_celline_info.loc[ccle_celline_info['CCLE name'] == 'TT_OESOPHAGUS', 'Cell line primary name']='T.T'
#     ccle_celline_mapping_dict = pd.Series(ccle_celline_info['Cell line primary name'].values,
#                                           index=ccle_celline_info['CCLE name']).to_dict()
#     return ccle_celline_mapping_dict


# def preprocess_gdsc_target_dat(gdsc1_file=data_config.gdsc_target_file1, gdsc2_file=data_config.gdsc_target_file2,
#                                score='AUC', output_file_path=None):
#     gdsc1_target_dat = pd.read_csv(gdsc1_file)[['CELL_LINE_NAME', 'DRUG_NAME', score]]
#     gdsc2_target_dat = pd.read_csv(gdsc2_file)[['CELL_LINE_NAME', 'DRUG_NAME', score]]
#
#     if score == 'LN_IC50':
#         gdsc1_target_dat[score] = np.exp(gdsc1_target_dat[score])
#         gdsc2_target_dat[score] = np.exp(gdsc2_target_dat[score])
#
#     gdsc1_target_dat = gdsc1_target_dat.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean()
#     gdsc2_target_dat = gdsc2_target_dat.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).mean()
#     gdsc1_target_dat = gdsc1_target_dat.loc[gdsc1_target_dat.index.difference(gdsc2_target_dat.index)]
#     target_dat = pd.concat([gdsc1_target_dat, gdsc2_target_dat], axis=0)
#     target_dat.reset_index(inplace=True)
#     target_dat['CELL_LINE_NAME'] = target_dat['CELL_LINE_NAME'].map(normalize_celline_names)
#
#     if score == 'LN_IC50':
#         target_dat[score] = np.log(target_dat[score])
#
#     if output_file_path:
#         target_dat.to_csv(output_file_path+'.csv')
#     return target_dat


def preprocess_target_data(score='AUC', output_file_path=None):
    # raw_target_df = preprocess_gdsc_target_dat(score=score, output_file_path=data_config.gdsc_target_file)
    # sample_list = list(get_normalized_ccle_sample_dict().values())
    # target_df = raw_target_df.pivot_table(index=['CELL_LINE_NAME'], columns='DRUG_NAME', values=score)
    # target_df = target_df[target_df.index.isin(sample_list)]

    #keep only tcga classified samples

    ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=4)
    ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()]
    ccle_sample_info.index = ccle_sample_info.index.astype('int')

    gdsc_sample_info = pd.read_csv(data_config.gdsc_sample_file, header=0, index_col=1)
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.index.dropna()]
    gdsc_sample_info.index = gdsc_sample_info.index.astype('int')
    gdsc_sample_info = gdsc_sample_info.loc[gdsc_sample_info.iloc[:, 8].dropna().index]

    gdsc_sample_mapping = gdsc_sample_info.merge(ccle_sample_info, left_index=True, right_index=True, how='inner')[
        ['Sample Name', 'DepMap_ID']]

    gdsc_sample_mapping.reset_index(inplace=True)
    gdsc_sample_mapping.set_index('Sample Name', inplace=True)
    gdsc_sample_mapping_dict = gdsc_sample_mapping.to_dict()['DepMap_ID']

    gdsc_drug_sensitivity = pd.read_csv(data_config.gdsc_target_file, skiprows=4, index_col=1)
    gdsc_drug_sensitivity.drop(axis=1, columns=[gdsc_drug_sensitivity.columns[0]], inplace=True)
    gdsc_drug_sensitivity.index = gdsc_drug_sensitivity.index.map(gdsc_sample_mapping_dict)
    target_df = gdsc_drug_sensitivity.loc[gdsc_drug_sensitivity.index.dropna()]
    target_df = target_df.astype('float32')
    if output_file_path:
        target_df.to_csv(output_file_path + '.csv', index_label='Sample')
    return target_df


def preprocess_ccle_gex_df(file_path=data_config.ccle_gex_file,
                           output_file_path=None):
    df = pd.read_csv(file_path, index_col=0)
    #get hgnc
    df.columns = df.columns.to_series().apply(lambda s: s[:s.find('(') - 1])

    # with gzip.open(filename=file_path) as f:
    #     df = pd.read_csv(f, sep='\t')
    # df.drop(columns=['transcript_ids'], inplace=True)
    # # could be done with pandas read, dtype converters
    # df.gene_id = df.gene_id.apply(lambda s: s[:s.find('.')])
    # df.set_index('gene_id', inplace=True)
    # df = df.transpose()
    # df.index.name = 'Sample'
    # df = df.loc[:, df.mean() != 0]
    # df = np.log2(df + 0.001)
    # normalized_name_dict = {name: name for name in df.index}
    # normalized_name_dict.update(get_normalized_ccle_sample_dict())
    # df.index = df.index.map(normalized_name_dict)
    # if feature_list:
    #     genes_to_keep = list(set(df.columns.tolist()) & set(feature_list))
    #     df = df[genes_to_keep]
    #     if output_file_path:
    #         output_file_path = output_file_path + '_filtered'
    df = df.astype('float32')
    print('Preprocessed data has {0} samples and {1} features'.format(df.shape[0], df.shape[1]))
    if output_file_path:
        df.to_csv(output_file_path + '.csv', index_label='Sample')
    return df

def load_ccle_muation_df(file_path=data_config.ccle_mut_file, filtered_variant=['Silent'], network_id_file=None):
    mutation_df = pd.read_csv(file_path, index_col=1)[['DepMap_ID', 'Variant_Classification']]
    mutation_df = mutation_df[~ mutation_df.Variant_Classification.isin(filtered_variant)]
    mutation_df.drop(columns=['Variant_Classification'], inplace=True)
    mutation_df['Score'] = 1
    binary_mutation_df = pd.pivot_table(data=mutation_df, columns='Hugo_Symbol', index='DepMap_ID', values='Score', fill_value=0,
                         aggfunc=max)

    if network_id_file:
        with gzip.open(network_id_file) as f:
            mapping_df = pd.read_csv(f, sep='\t', index_col=0)
        binary_mutation_df = binary_mutation_df[binary_mutation_df.columns.intersection(mapping_df.preferred_name)]

    print('Preprocessed data has {0} samples and {1} features'.format(binary_mutation_df.shape[0],
                                                                      binary_mutation_df.shape[1]))

    return binary_mutation_df


def preprocess_ccle_mut(propagation_flag=True, mutation_dat_file=data_config.ccle_mut_file,
                   kernel_file=data_config.propagation_kernel_file,network_id_file=data_config.string_id_mapping_file,
                   output_file_path=None):
    if propagation_flag:
        binary_mutation_df = load_ccle_muation_df(mutation_dat_file, network_id_file=network_id_file)
        print('Propagation Start!')

        #kernel = pd.read_feather(kernel_file, columns=binary_mutation_df.columns.union(['index']))
        #kernel = pd.read_feather(kernel_file)
        kernel = feather.read_dataframe(kernel_file)
        kernel.set_index('index', inplace=True)
        kernel.index = [ind.decode() for ind in kernel.index]
        kernel = kernel.transpose()
        binary_mutation_df = binary_mutation_df[binary_mutation_df.columns.intersection(kernel.columns)]
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
            print('Preprocessed data has {0} samples and {1} features'.format(propagation_result.shape[0],
                                                                              propagation_result.shape[1]))
            propagation_result.to_csv(output_file_path + '_propagated.csv', index_label='Sample')
        return propagation_result
    else:
        binary_mutation_df = load_ccle_muation_df(mutation_dat_file)
        if output_file_path:
            print('Preprocessed data has {0} samples and {1} features'.format(binary_mutation_df.shape[0],
                                                                              binary_mutation_df.shape[1]))
            binary_mutation_df.to_csv(output_file_path + '.csv', index_label='Sample')
        return binary_mutation_df


if __name__ == '__main__':
    #preprocess_ccle_gex_df(output_file_path=data_config.ccle_preprocessed_gex_file)
    preprocess_ccle_mut(output_file_path=data_config.ccle_preprocessed_mut_file)
    preprocess_target_data(output_file_path=data_config.gdsc_preprocessed_target_file)
