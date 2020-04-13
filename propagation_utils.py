# to be run in python 2.7 with pyNBS installed

import data_config
import numpy as np
import pandas as pd
from pyNBS import data_import_tools as dit
from pyNBS import network_propagation as prop


def preprocess_network_file(raw_network_file_path=data_config.raw_string_network_file,
                            mapping_file_path=data_config.string_id_mapping_file,
                            output_file_path=data_config.current_network_file, edge_percentile=.9):
    #need to be modified accordingly
    with gzip.open(mapping_file_path) as f:
        mapping_df = pd.read_csv(f, sep='\t', index_col=0)
    with gzip.open(raw_network_file_path) as f:
        network_df = pd.read_csv(f, sep=' ')
        network_df.drop_duplicates(inplace=True)
    network_df = \
        mapping_df.merge(network_df, left_index=True, right_on='protein1').merge(mapping_df, left_on='protein2',
                                                                                 right_index=True)[
            ['preferred_name_x', 'preferred_name_y', 'combined_score']]
    if edge_percentile < 1.:
        network_df = network_df.loc[network_df.iloc[:, 2] >= network_df.iloc[:, 2].quantile(edge_percentile), :]

    network_df = network_df.iloc[:, :2]
    network_df.drop_duplicates(inplace=True)
    network_features = list(set(network_df.iloc[:, 0].tolist()) | set(network_df.iloc[:, 1].tolist()))

    print('Preprocessed network has {0} different features and {1} edges'.format(len(network_features),
                                                                                 network_df.shape[0]))

    if output_file_path:
        network_df.to_csv(output_file_path, header=False, index=False, sep='\t')
    else:
        return network_df


def generate_precompute_kernel(network_file_path, output_file_path, alpha=0.7):
    network = dit.load_network_file(network_file_path)
    network_nodes = network.nodes()
    network_I = pd.DataFrame(np.identity(len(network_nodes)), index=network_nodes, columns=network_nodes)

    kernel = prop.network_propagation(network, network_I, alpha=alpha, symmetric_norm=False)
    kernel.columns = [str(col) for col in kernel.columns]
    kernel.index = [str(ind) for ind in kernel.index]
    kernel.sort_index(inplace=True)
    kernel.sort_index(axis=1, inplace=True)

    # assert all(kernel.index == kernel.columns)
    kernel = kernel.transpose()
    # assert all(kernel.sum()-1 < 1e-10)
    kernel.reset_index(inplace=True)
    try:
        kernel.to_feather(output_file_path)
    except ValueError:
        kernel.to_csv(output_file_path)
    return output_file_path


if __name__ == '__main__':
    generate_precompute_kernel()
