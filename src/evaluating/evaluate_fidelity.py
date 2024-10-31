import pandas as pd
from itertools import combinations

from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_table import LogisticDetection

from src.evaluating.metrics_fidelity import compute_WD, compute_JSD
from src.evaluating.metrics_fidelity import compute_TVComplement, compute_KSComplement
from src.evaluating.metrics_fidelity import compute_CorrelationSimilarity
from src.evaluating.metrics_fidelity import compute_ContingencySimilarity


def evaluate_fidelity(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    metadata: SingleTableMetadata(),
    list_metrics: list = [
        "WD",
        "JSD",
        "KSComplement",
        "TVComplement",
        "CorrelationSimilarity",
        "ContingencySimilarity",
    ],
) -> dict:
    """Takes the real and the synthetic datasets and compares them according to various metrics. Metrics available : WD (Wasserstein Distance), JSD (Jensen Shannon Divergence), KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity

    Args:
        df_real (pd.DataFrame): Dataset with real data
        df_synth (pd.DataFrame): Dataset with synthetic data
        list_metrics (list): List with the metrics the user wants to compute. Must be a subset of ["WD","JSD","KSComplement","TVComplement","CorrelationSimilarity", "ContingencySimilarity"]. Default is all metrics.

    Returns:
        dict: Dictionnary with the metrics names as keys and the metrics values as values.
    """
    primary_key = metadata.to_dict().get("primary_key")
    # remove primary key to compute metrics
    # necessary not for sdv metrics but for custom metrics such as wD and JSD
    df_real_ = df_real.drop(columns=primary_key)
    df_synth_ = df_synth.drop(columns=primary_key)
    
    dict_results = {}
    for metric in list_metrics:
        if metric == "WD":
            wd = compute_WD(df_real_, df_synth_)
            dict_results[metric] = wd

        elif metric == "JSD":
            # remove primary key to compute metrics
            # necessary not for sdv metrics but for custom metrics such as wD and JSD
            
            jsd = compute_JSD(df_real_, df_synth_)
            dict_results[metric] = jsd

        elif metric == "KSComplement":
            ks = compute_KSComplement(df_real, df_synth, metadata)
            dict_results[metric] = ks

        elif metric == "TVComplement":
            tv = compute_TVComplement(df_real, df_synth, metadata)
            dict_results[metric] = tv

        elif metric == "CorrelationSimilarity":
            corrnum = compute_CorrelationSimilarity(df_real, df_synth, metadata)
            dict_results[metric] = corrnum

        elif metric == "ContingencySimilarity":
            corrdisc = compute_ContingencySimilarity(df_real, df_synth, metadata)
            dict_results[metric] = corrdisc
        
        elif metric == "LogisticDetection":
            logd = LogisticDetection.compute(
                    real_data=df_real,
                    synthetic_data=df_synth,
                    metadata=metadata
                )
            dict_results[metric] = logd
            
        else:
            raise ValueError(f"{metric} is not a valid metric")
   
    return dict_results

def evaluate_correlations(
    df_real: pd.DataFrame, df_synth: pd.DataFrame, list_continuous_columns: list
) -> pd.DataFrame:
    """Generates correlation scores on continuous columns

    Args:
        df_real (pd.DataFrame): Dataframe with real data
        df_synth (pd.DataFrame): Dataframe with synthetised data
        list_continuous_columns (list): Name of continuous columns

    Returns:
        pd.DataFrame: Dataframe with correlation similarity for each combination of columns
    """
    df_corr = pd.DataFrame()
    col_combinations = combinations(list_continuous_columns, 2)
    for col1, col2 in col_combinations:
        df_corr.loc[
            "correlation_score", f"{col1}_{col2}"
        ] = CorrelationSimilarity.compute(df_real[[col1, col2]], df_synth[[col1, col2]])
    return df_corr


def get_score_plot(sdv_report: QualityReport):
    """Retrieves a plotly score plot"""
    score_plot = (
        sdv_report.get_visualization(property_name="Column Shapes")
        # format xaxis
        .update_xaxes(tickfont_size=10, tickangle=-30).update_yaxes(tickfont_size=10)
        # format legend (to gain space and readibility)
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1,
            ),
        )
    )
    return score_plot


def get_correlation_plot(sdv_report: QualityReport):
    """Retrieves a plotly correlation plot"""
    corr_plot = (
        sdv_report.get_visualization(property_name="Column Pair Trends")
        # format xaxis
        .update_xaxes(tickfont_size=10, tickangle=-30).update_yaxes(tickfont_size=10)
        # format legend (to gain space and readibility)
        .update_layout(
            legend=dict(
                orientation="h",
                xanchor="right",
                x=1,
                yanchor="bottom",
                y=1,
            ),
        )
    )
    return corr_plot


