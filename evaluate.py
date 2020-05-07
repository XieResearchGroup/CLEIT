import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from collections import defaultdict

def drug_wise_evaluation(truth_df, pred_df):
    pred_df = pred_df.loc[truth_df.index]
    per_drug_measurement = defaultdict(dict)
    for drug in truth_df.columns:
        samples = truth_df.index[~truth_df[drug].isna()]
        per_drug_measurement['pearson'][drug] = pearsonr(truth_df.loc[samples, drug], pred_df.loc[samples, drug])[0]
        per_drug_measurement['spearman'][drug] = spearmanr(truth_df.loc[samples, drug], pred_df.loc[samples, drug])[0]
        per_drug_measurement['rmse'][drug] = sqrt(
            mean_squared_error(truth_df.loc[samples, drug], pred_df.loc[samples, drug]))
        per_drug_measurement['r2'][drug] = r2_score(truth_df.loc[samples, drug], pred_df.loc[samples, drug])

    return pd.DataFrame.from_dict(per_drug_measurement)

def cell_wise_evaluation(truth_df, pred_df):
    pred_df = pred_df.loc[truth_df.index]
    per_cell_measurement = defaultdict(dict)
    k_vecs = [1, 3, 5, 10]
    for k in k_vecs:
        for cell in truth_df.index:
            drugs = truth_df.columns[~truth_df.loc[cell].isna()]
            if len(drugs) < k:
                per_cell_measurement[k][cell] = None

            else:
                topk_truth = truth_df.loc[cell, drugs].nlargest(k).index
                topk_pred = pred_df.loc[cell,].nlargest(k).index
                per_cell_measurement[k][cell] = len(topk_truth.intersection(topk_pred)) / k

    return pd.DataFrame.from_dict(per_cell_measurement)




