import pandas as pd

from sdv.metadata import SingleTableMetadata
from src.evaluating.metrics_privacy import compute_NewRowSynthesis
from src.evaluating.metrics_privacy import compute_DCR
from src.evaluating.metrics_privacy import compute_NNDR
from src.evaluating.metrics_privacy import compute_CategoricalCAP

def evaluate_privacy(df_real: pd.DataFrame,
                    df_synth: pd.DataFrame,
                    metadata: SingleTableMetadata(),
                    list_metrics: list = ["NewRowSynthesis",
                                          "DCR",
                                          "NNDR"],
                    sample_size: float=0.1,
                    sensitive_fields: list=None,
                    key_fields: list=None) -> dict:
    """Takes the real and the synthetic datasets and compares them according to various metrics. Metrics available : WD (Wasserstein Distance), JSD (Jensen Shannon Divergence), KSComplement, TVComplement, CorrelationSimilarity, ContingencySimilarity

    Args:
        df_real (pd.DataFrame): Dataset with real data
        df_synth (pd.DataFrame): Dataset with synthetic data
        sdv_metadata (SingleTableMetadata()): sdv metadata dict
        sample_size (pd.DataFrame): sample size of the synthetic data to select to compute privacy metrics 
        list_metrics (list): List with the metrics the user wants to compute. Must be a subset of ["NewRowSynthesis","DCR","NNDR"]. Default is all metrics.

    Returns:
        dict: Dictionnary with the metrics names as keys and the metrics values as values.
    """
     # remove primary key to compute metrics
    # necessary not for sdv metrics but for custom metrics such as wD and JSD
    primary_key = metadata.to_dict().get("primary_key")
    df_real_ = df_real.copy().drop(columns=primary_key)
    df_synth_ = df_synth.copy().drop(columns=primary_key)
    dict_results = {}
    for metric in list_metrics:

        if metric == "NewRowSynthesis":
            nrs = compute_NewRowSynthesis(df_real=df_real,
                                          df_synth=df_synth,
                                          metadata=metadata,
                                          synthetic_sample_size=sample_size)
            dict_results[metric] = nrs

        elif metric == "DCR":
            dcr = compute_DCR(df=df_real_,
                              other_df=df_synth_,
                              sample_size=sample_size)
            dict_results[metric] = dcr

        elif metric == "NNDR":
            nndr = compute_NNDR(df=df_real_,
                              other_df=df_synth_,
                              sample_size=sample_size)
            dict_results[metric] = nndr
        
        elif metric == 'CategoricalCAP':  
            assert sensitive_fields is not None, "sensitive_fields must be provided \
                to compute CategoricalCAP" 
            assert sensitive_fields is not None, "key_fields must be provided \
            to compute CategoricalCAP" 
            list_cols = metadata.get_column_names()  
            list_cols.remove(primary_key)   
            ccap = compute_CategoricalCAP(df_real=df_real_,
                                          df_synth=df_synth_,
                                          key_fields=key_fields,
                                          sensitive_fields=sensitive_fields)
            dict_results[metric] = ccap.values.mean()
        else:
            raise ValueError(f"Metric {metric} is not available")

    return dict_results



