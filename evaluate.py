import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from collections import defaultdict

def drug_wise_evaluation(truth_df, pred_df):
    pred_df = pred_df.loc[truth_df.index]
    per_drug_measurement = defaultdict(dict)
    for drug in truth_df.columns:
        samples = truth_df.index[~truth_df[drug].isna()]
        if len(samples) <=1:
            per_drug_measurement['pearson'][drug] = np.nan
            per_drug_measurement['spearman'][drug] = np.nan
            per_drug_measurement['rmse'][drug] = np.nan
            per_drug_measurement['r2'][drug] = np.nan
        else:
            per_drug_measurement['pearson'][drug] = pearsonr(truth_df.loc[samples, drug], pred_df.loc[samples, drug])[0]
            per_drug_measurement['spearman'][drug] = spearmanr(truth_df.loc[samples, drug], pred_df.loc[samples, drug])[
                0]
            per_drug_measurement['rmse'][drug] = sqrt(
                mean_squared_error(truth_df.loc[samples, drug], pred_df.loc[samples, drug]))
            per_drug_measurement['r2'][drug] = r2_score(truth_df.loc[samples, drug], pred_df.loc[samples, drug])

    return pd.DataFrame.from_dict(per_drug_measurement)

def cell_wise_evaluation(truth_df, pred_df):
    pred_df = pred_df.loc[truth_df.index]
    per_cell_measurement = defaultdict(dict)
    for cell in truth_df.index:
        drugs = truth_df.columns[~truth_df.loc[cell].isna()]
        if len(drugs) <= 1:
            per_cell_measurement['pearson'][cell] = np.nan
            per_cell_measurement['spearman'][cell] = np.nan
            per_cell_measurement['rmse'][cell] = np.nan
            per_cell_measurement['r2'][cell] = np.nan
        else:
            per_cell_measurement['pearson'][cell] = pearsonr(truth_df.loc[cell, drugs], pred_df.loc[cell, drugs])[0]
            per_cell_measurement['spearman'][cell] = spearmanr(truth_df.loc[cell, drugs], pred_df.loc[cell, drugs])[0]
            per_cell_measurement['rmse'][cell] = sqrt(
                mean_squared_error(truth_df.loc[cell, drugs], pred_df.loc[cell, drugs]))
            per_cell_measurement['r2'][cell] = r2_score(truth_df.loc[cell, drugs], pred_df.loc[cell, drugs])
    return pd.DataFrame.from_dict(per_cell_measurement)

def cell_wise_top_k_evaluation(truth_df, pred_df):
    pred_df = pred_df.loc[truth_df.index]
    per_cell_measurement = defaultdict(dict)
    k_vecs = [1, 3, 5, 10]
    for k in k_vecs:
        for cell in truth_df.index:
            drugs = truth_df.columns[~truth_df.loc[cell].isna()]
            if len(drugs) <= k:
                per_cell_measurement[k][cell] = None
            else:
                topk_value = truth_df.loc[cell, drugs].nlargest(k)[-1]
                num_drugs = (truth_df.loc[cell, drugs] >= topk_value).sum()
                topk_truth = truth_df.loc[cell,drugs][truth_df.loc[cell,drugs]>=topk_value].index
                pred_topk_value = pred_df.loc[cell,].nlargest(k)[-1]
                topk_pred = pred_df.loc[cell,][pred_df.loc[cell,]>=pred_topk_value].index
                per_cell_measurement[k][cell] = len(topk_truth.intersection(topk_pred)) / num_drugs

    return pd.DataFrame.from_dict(per_cell_measurement)


def bootstrap_baseline_random_results_drug_wise(truth_df, B=1000):
    measurement_df = None
    for _ in range(B):
        pred_df = pd.DataFrame(np.random.uniform(size=truth_df.shape),
                               index=truth_df.index,
                               columns=truth_df.columns)
        temp_df = drug_wise_evaluation(truth_df=truth_df, pred_df=pred_df)

        if measurement_df is None:
            measurement_df=temp_df.mean()
        else:
            measurement_df = pd.concat([measurement_df, temp_df.mean()], axis=1)

    random_guess_base_df = pd.DataFrame(np.full_like(truth_df, fill_value=-1),
                                        index=truth_df.index,
                                        columns=truth_df.columns)
    random_guess_base_df = random_guess_base_df.assign(**truth_df.mean().to_dict())
    return measurement_df.mean(axis=1), drug_wise_evaluation(truth_df=truth_df, pred_df=random_guess_base_df).mean()



def bootstrap_baseline_random_results_cell_wise(truth_df, B=100):
    measurement_df = None
    for _ in range(B):
        pred_df = pd.DataFrame(np.random.uniform(size=truth_df.shape),
                               index=truth_df.index,
                               columns=truth_df.columns)
        temp_df = cell_wise_evaluation(truth_df=truth_df, pred_df=pred_df)

        if measurement_df is None:
            measurement_df=temp_df.mean()
        else:
            measurement_df = pd.concat([measurement_df, temp_df.mean()], axis=1)

    # random_guess_base_df = pd.DataFrame(np.full_like(truth_df, fill_value=-1),
    #                                     index=truth_df.index,
    #                                     columns=truth_df.columns)
    random_guess_base_df = truth_df.transpose().assign(**truth_df.mean(axis=1).to_dict()).transpose()
    return measurement_df.mean(axis=1), cell_wise_evaluation(truth_df=truth_df, pred_df=random_guess_base_df).mean()

def topk_precision_bar_plot(plot_df, output_file_path='topk_precision_placehold.png'):
    g = sns.catplot(data=plot_df, col='k', col_wrap=2, x='method', y='precision',kind='bar')
    g.savefig(output_file_path)









