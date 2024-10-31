import pandas as pd
import numpy as np

from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import BinaryAdaBoostClassifier
from sdmetrics.single_table import BinaryDecisionTreeClassifier


def evaluate_utility(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    metadata: SingleTableMetadata(),
    col_target: str,
    list_metrics: list = ["BinaryAdaBoostClassifier",
                          "BinaryDecisionTreeClassifier"],
) -> dict:
    """Takes the real and the synthetic datasets and compares them according to various metrics. Metrics available : WD (Wasserstein Distance), JSD (Jensen Shannon Divergence), KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity

    Args:
        df_real (pd.DataFrame): Dataset with real data
        df_synth (pd.DataFrame): Dataset with synthetic data
        col_target (str: A string representing the name of the column that you want to predict. This must be a boolean column.)
        list_metrics (list): List with the metrics the user wants to compute. Must be a subset of ["BinaryAdaBoostClassifier",
                          "BinaryDecisionTreeClassifier"]. Default is all metrics.
    Returns:
        dict: Dictionnary with the metrics names as keys and the metrics values as values.
    """
    
    dict_results = {}   
    primary_key = metadata.to_dict().get("primary_key")     
    try:
        metadata.add_column(column_name=f'{col_target}_bin',
                        sdtype='binary')
    except:
        pass

    # TODO remove if binary column
    df_real = create_bin_col(df_real, col_target)
    df_synth = create_bin_col(df_synth, col_target)
    
    # necessity to have PTID in numerical format for this metrics (SDV bug)
    df_real[primary_key] = np.arange(0, df_real.shape[0])
    df_synth[primary_key] = np.arange(0, df_synth.shape[0])
    
    for metric in list_metrics:
        if metric == "BinaryAdaBoostClassifier":
            f1score_score = BinaryAdaBoostClassifier.compute(
                    test_data=df_real,
                    train_data=df_synth,
                    target=f"{col_target}_bin",
                    metadata=metadata)
            dict_results[metric] = f1score_score
    return dict_results  

def create_bin_col(df: pd.DataFrame,
                   col: str) -> pd.DataFrame:
    # add a binary target variable
    median_value = df[col].median()
    df[f"{col}_bin"] = df[col].apply(lambda x: 1 if x >= median_value else 0)
    return df