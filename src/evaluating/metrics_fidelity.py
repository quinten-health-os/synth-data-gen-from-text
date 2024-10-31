import pandas as pd
import numpy as np
import logging 
from itertools import combinations

from sdv.metadata import SingleTableMetadata
from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def compute_WD(df_real: pd.DataFrame,
               df_synth: pd.DataFrame) -> float:
    """Computes Wasserstein Distance to compare two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real dataset

    Returns:
        float: Mean of the Wasserstein Distance scores. Varies between [0,inf[. The greater the score, the more similar the data.
    """
    wd_col = []
    for col in df_real.columns:
        wd_col.append(wasserstein_distance(df_real[col], df_synth[col]))
    wd = np.mean(wd_col)
    return wd


def compute_JSD(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Computes Jensen Shannon Divergence to compare two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real dataset

    Returns:
        float: Mean of the Jensen Shannon scores. Varies between [0, ln(2)].
        The closer from 0 the more similar the data.
        Value can diverge if some events have a null probability of happening.
    """
    if len(df_real) != len(df_synth):
        k_sample = min(len(df_real), len(df_synth))
        df_real = df_real.sample(k_sample)
        df_synth = df_synth.sample(k_sample)
    jsd_all = jensenshannon(df_real, df_synth, axis=1)
    jsd_mean = np.mean(jsd_all)
    if jsd_mean == np.inf:
        logging.info(
            "Events with a zero probability of happening caused infinite divergence while computing Jensen Shannon Divergence"
        )
    return jsd_mean


def compute_KSComplement(df_real: pd.DataFrame,
                         df_synth: pd.DataFrame,
                         metadata: SingleTableMetadata()) -> float:
    """Computes KSComplement score to compare the continuous variables distributions from two dataframes.

    Args:
         df_real (pd.DataFrame): Real dataset
         df_synth (pd.DataFrame): Dataset synthesized from real dataset

     Returns:
         float: Mean of the KSComplement scores. Varies between [0,1]. The closer to 1 the more similar the data.
    """
    continuous_columns = metadata.get_column_names(sdtype='numerical')
    ks_all = []
    for col in continuous_columns:
        ks = KSComplement.compute(
            real_data=df_real[col], synthetic_data=df_synth[col]
        )
        ks_all.append(ks)
    ks_mean = np.mean(ks_all)
    return ks_mean


def compute_TVComplement(df_real: pd.DataFrame,
                         df_synth: pd.DataFrame,
                         metadata: SingleTableMetadata) -> float:
    """Computes TVComplement score to compare the discrete variables distributions from two dataframes.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the TVComplement scores. Varies between [0,1]. The closer to 1 the more similar the data.
    """
    discrete_columns = metadata.get_column_names(sdtype='categorical')
    discrete_columns += metadata.get_column_names(sdtype='boolean')

    tv_all = []
    for col in discrete_columns:
        tv = TVComplement.compute(
            real_data=df_real[col], synthetic_data=df_synth[col]
        )
        tv_all.append(tv)
    tv_mean = np.mean(tv_all)
    return tv_mean


def compute_CorrelationSimilarity(df_real: pd.DataFrame,
                                  df_synth: pd.DataFrame,
                                  metadata: SingleTableMetadata()
) -> float:
    """Computes a correlation score to compare correlations of continuous variables in a real dataset vs in a synthesized dataset.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the CorrelationSimilarity scores. Varies between [0,1]. The closer to 1 the more similar the correlations between the two datasets.
    """
    continuous_columns = continuous_columns = metadata.get_column_names(sdtype='numerical')
    cont_col_combinations = combinations(continuous_columns, 2)
    corrnum_all = []
    for (col1, col2) in cont_col_combinations:
        corrnum = CorrelationSimilarity.compute(
            df_real[[col1, col2]], df_synth[[col1, col2]]
        )
        corrnum_all.append(corrnum)
    corrnum_mean = np.mean(corrnum_all)
    return corrnum_mean


def compute_ContingencySimilarity(df_real: pd.DataFrame,
                                  df_synth: pd.DataFrame,
                                  metadata: SingleTableMetadata()
) -> float:
    """Computes a correlation score to compare correlations of discrete variables in a real dataset vs in a synthesized dataset.

    Args:
        df_real (pd.DataFrame): Real dataset
        df_synth (pd.DataFrame): Dataset synthesized from real datase

    Returns:
        float: Mean of the ContingencySimilarity scores. Varies between [0,1]. The closer to 1 the more similar the correlations between the two datasets.
    """
    discrete_columns = metadata.get_column_names(sdtype='categorical')
    discrete_columns += metadata.get_column_names(sdtype='boolean')
    
    disc_col_combinations = combinations(discrete_columns, 2)
    corrdisc_all = []
    for (col1, col2) in disc_col_combinations:
        corrdisc = ContingencySimilarity.compute(
            df_real[[col1, col2]], df_synth[[col1, col2]]
        )
        corrdisc_all.append(corrdisc)
    corrdisc_mean = np.mean(corrdisc_all)
    return corrdisc_mean


