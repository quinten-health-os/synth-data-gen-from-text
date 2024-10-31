import pandas as pd
import numpy as np
import logging

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sdmetrics.single_table import CategoricalCAP, NewRowSynthesis
from sdv.metadata import SingleTableMetadata

def compute_NewRowSynthesis(
        df_real: pd.DataFrame,
        df_synth: pd.DataFrame,
        metadata: SingleTableMetadata,
        numerical_match_tolerance: float=0.01,
        synthetic_sample_size: float=None,
) -> float:
    """ Computes the proportion of new rows in synthetic data.

    Args:
        df_real (pd.DataFrame): Dataframe with real data.
        df_synth (pd.DataFrame): Dataframe with synthetized data.
        primary_key (str): Primary key (often UUID) to create metadata dictionnary
        numerical_match_tolerance (float): Float >0 representing how close two numerical values have to be in order to be considered a match. Default is 0.01.
        synthetic_sample_size (float): Percentage of synthetic rows to sample before computing the metric. Helps speed up the computation time if needed. Default is None.

    Returns:
        float: _description_
    """
    if synthetic_sample_size is not None:
        sample_size = int(len(df_real)*(synthetic_sample_size))
        logging.info(f"Synthetic data sample size: {sample_size}")

    else:
        sample_size =  None
    score = NewRowSynthesis.compute(df_real, df_synth, metadata, numerical_match_tolerance, sample_size)
    return score


def scale_dataframe(df: pd.DataFrame,
                    sample_size: float=0.1)->pd.DataFrame:
    """ Scales a dataframe and samples it if needed. Code source from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py

    Args:
        df (pd.DataFrame): Dataframe to scale
        sample_size (float, optional): Percentage of the dataframe to sample. Defaults to None.

    Returns:
        pd.DataFrame: Scaled (and possibly sampled) dataframe.
    """
    df = df.drop_duplicates(keep=False)
    if sample_size is not None:
        df = df.sample(n=int(len(df)*(sample_size)), random_state=42).to_numpy()

    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    return df_scaled

def compute_smallest_distances(df: pd.DataFrame,
                               other_df: pd.DataFrame=None)->np.array:
    """ Computes the two smallest distances between datasets. By default computes it within one dataset but can be done between two different ones. Computing it within one dataset allows to appoint a threshold when comparing two different ones. Code source from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py

    Args:
        df (pd.DataFrame): Base dataset to compute distance on.
        sample_size (float, optional):  Percentage of the dataframe to sample. Defaults to None.
        other_df (pd.DataFrame, optional): Other dataframe to compare to the base one if needed. Defaults to None.

    Returns:
        np.array: The two smallest distances computed between dataframes.
    """
    # Compute pair-wise distances between real and synthetic, real and real, synthetic and synthetic
    dist = metrics.pairwise_distances(df, Y=other_df, metric="minkowski", n_jobs=-1)

    if other_df is None:
        # Remove distances of data points to themselves to avoid 0s
        rd_dist = dist[~np.eye(dist.shape[0],dtype=bool)].reshape(dist.shape[0],-1)
    else: 
        rd_dist = dist

    # Computing first and second smallest nearest neighbour distances
    smallest_two_indexes = [rd_dist[i].argsort()[:2] for i in range(len(rd_dist))]
    smallest_two_distances = [rd_dist[i][smallest_two_indexes[i]] for i in range(len(rd_dist))]       
    return smallest_two_distances


def compute_DCR(df: pd.DataFrame,
                sample_size: float=0.1,
                other_df: pd.DataFrame=None)->float:
    """ Computes the fifth percentile of the distance to closest record. The higher the better. Code source from
    https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py

    Args:
        df (pd.DataFrame): Base dataset to compute distance on.
        other_df (pd.DataFrame, optional): Other dataframe to compare to the base one if needed. Defaults to None.
        sample_size (float, optional): Percentage of the dataframe to sample. Defaults to 0.1.

    Returns:
        float: Fifth percentile of the DCR.
    """
    df_scaled = scale_dataframe(df, sample_size)
    other_df_scaled = None
    if other_df is not None:
        other_df_scaled = scale_dataframe(other_df, sample_size)

    smallest_distances = compute_smallest_distances(df_scaled, other_df_scaled)
    min_dist = np.array([i[0] for i in smallest_distances])
    fifth_perc_dcr = np.percentile(min_dist,5)
    return fifth_perc_dcr


def compute_NNDR(df: pd.DataFrame,
                 sample_size: float=0.1,
                 other_df: pd.DataFrame=None)->float:
    """ Computes the fifth percentile of the Nearest Neighbor Distance Ratio. It measures the proximity to outlers in the original dataset.
    The closer to 1 the better. Code source from https://github.com/Team-TUD/CTAB-GAN/blob/main/model/eval/evaluation.py

    Args:
        df (pd.DataFrame): Base dataset to compute distance on.
        other_df (pd.DataFrame, optional): Other dataframe to compare to the base one if needed. Defaults to None.
        sample_size (float, optional): Percentage of the dataframe to sample. Defaults to None.

    Returns:
        float: Fifth percentile of the NNDR.
    """
    df_scaled = scale_dataframe(df, sample_size)
    other_df_scaled = None
    if other_df is not None:
        other_df_scaled = scale_dataframe(other_df, sample_size)

    smallest_distances = compute_smallest_distances(df_scaled, other_df_scaled)
    nn_ratio = np.array([i[0]/i[1] for i in smallest_distances])
    nn_fifth_perc = np.percentile(nn_ratio, 5)
    return nn_fifth_perc

### Privacy attack metrics ###

def compute_CategoricalCAP(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    key_fields: list,
    sensitive_fields: list,
) -> pd.DataFrame:
    """Computes CAP privacy score considering an attacker wants to guess one of the sensitives fields columns and knows the key fields columns.

    Args:
        df_real (pd.DataFrame): Dataframe with real data.
        df_synth (pd.DataFrame): Dataframe with synthetized data.
        key_fields (list): Columns known by the attacker.
        sensitive_fields (list): Column the attacker tries to guess.

    Returns:
        pd.DataFrame: Dataframe with privacy scores for each sensitive column.
    """
    df_scores = pd.DataFrame()
    for sensitive_col in sensitive_fields:
        score = CategoricalCAP.compute(
            real_data=df_real,
            synthetic_data=df_synth,
            key_fields=key_fields,
            sensitive_fields=[sensitive_col],
        )
        df_scores.loc["CategoricalCAP", sensitive_col] = score
    return df_scores