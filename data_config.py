import os
"""
configuration file includes all related multi-omics datasets 
"""

root_data_folder = './data'
raw_data_folder = os.path.join(root_data_folder, 'raw_dat')
preprocessed_data_folder = os.path.join(root_data_folder, 'preprocessed_dat')
gex_feature_file = os.path.join(preprocessed_data_folder, 'uq1000_gex_feature.csv')
xena_mut_uq_file = os.path.join(preprocessed_data_folder, 'xena_uq_mut_standarized.csv')
ccle_mut_uq_file = os.path.join(preprocessed_data_folder, 'ccle_uq_mut_standarized.csv')

#mapping_file = os.path.join(raw_data_folder, 'mart_export.txt')
gene_feature_file = os.path.join(preprocessed_data_folder, 'CosmicHGNC_list.tsv')
#Xena datasets
xena_folder = os.path.join(raw_data_folder, 'Xena')
xena_id_mapping_file = os.path.join(xena_folder, 'gencode.v23.annotation.gene.probemap')
xena_gex_file = os.path.join(xena_folder, 'tcga_RSEM_gene_tpm.gz')
xena_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'xena_gex')
xena_mut_file = os.path.join(xena_folder, 'mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz')
xena_preprocessed_mut_file = os.path.join(preprocessed_data_folder, 'xena_mut')
xena_sample_file = os.path.join(xena_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'CCLE_expression.csv')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'ccle_gex')
ccle_mut_file = os.path.join(ccle_folder, 'CCLE_mutations.csv')
ccle_preprocessed_mut_file = os.path.join(preprocessed_data_folder, 'ccle_mut')
ccle_sample_file = os.path.join(ccle_folder, 'sample_info.csv')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_target_file = os.path.join(gdsc_folder, 'sanger-dose-response.csv')
gdsc_sample_file = os.path.join(gdsc_folder, 'gdsc_cell_line_annotation.csv')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_target')

#PPI network files
network_folder = os.path.join(raw_data_folder, 'network')
string_network_folder = os.path.join(network_folder, 'STRING')
raw_string_network_file = os.path.join(string_network_folder, '9606.protein.links.v11.0.txt.gz')
string_id_mapping_file = os.path.join(string_network_folder, '9606.protein.info.v11.0.txt.gz')
current_network_file = os.path.join(string_network_folder, 'string_network_hgnc.txt')
propagation_kernel_file = os.path.join(string_network_folder, 'string_propagation_kernel.file')