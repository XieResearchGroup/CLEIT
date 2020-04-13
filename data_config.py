import os
"""
configuration file includes all related multi-omics datasets 
"""

raw_data_folder = './dat/raw_dat'
preprocessed_data_folder = './dat/preprocessed_dat'

#Xena datasets
xena_folder = os.path.join(raw_data_folder, 'Xena')
xena_id_mapping_file = os.path.join(raw_data_folder, 'gencode.v23.annotation.gene.probemap')
xena_gex_file = os.path.join(xena_folder, 'tcga_RSEM_gene_tpm.gz')
xena_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'xena_gex')
xena_mut_file = os.path.join(xena_folder, 'mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz')
xena_preprocessed_mut_file = os.path.join(preprocessed_data_folder, 'xena_mut')
xena_cnv_file = os.path.join(xena_folder, 'Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz')
xena_sample_file = os.path.join(xena_folder, 'TCGA_phenotype_denseDataOnlyDownload.tsv.gz')

#CCLE datasets
ccle_folder = os.path.join(raw_data_folder, 'CCLE')
ccle_gex_file = os.path.join(ccle_folder, 'CCLE_RNAseq_rsem_genes_tpm_20180929.txt.gz')
ccle_preprocessed_gex_file = os.path.join(preprocessed_data_folder, 'ccle_gex')
ccle_mut_file = os.path.join(ccle_folder, 'CCLE_DepMap_18q3_maf_20180718.txt')
ccle_preprocessed_mut_file = os.path.join(preprocessed_data_folder, 'ccle_mut')
ccle_cnv_file = os.path.join(ccle_folder, 'CCLE_copynumber_byGene_2013-12-03.txt')
ccle_sample_file = os.path.join(ccle_folder, 'CCLE_sample_info_file_2012-10-18.txt')

#GDSC datasets
gdsc_folder = os.path.join(raw_data_folder, 'GDSC')
gdsc_target_file1 = os.path.join(gdsc_folder, 'GDSC1_fitted_dose_response_25Feb20.csv')
gdsc_target_file2 = os.path.join(gdsc_folder, 'GDSC2_fitted_dose_response_25Feb20.csv')
gdsc_target_file = os.path.join(preprocessed_data_folder, 'gdsc_raw_target')
gdsc_preprocessed_target_file = os.path.join(preprocessed_data_folder, 'gdsc_target')

#PPI network files
network_folder = os.path.join(raw_data_folder, 'network')
string_network_folder = os.path.join(network_folder, 'STRING')
raw_string_network_file = os.path.join(string_network_folder, '9606.protein.links.v11.0.txt.gz')
string_id_mapping_file = os.path.join(string_network_folder, '9606.protein.info.v11.0.txt.gz')
current_network_file = os.path.join(string_network_folder, 'string_network_hgnc.txt')
propagation_kernel_file = os.path.join(string_network_folder, 'string_propagation_kernel.file')