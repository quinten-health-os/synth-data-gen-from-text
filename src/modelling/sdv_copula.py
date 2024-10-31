import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

def fit_copula(df: pd.DataFrame, sdv_metadata: dict):
    # init metadata
    synthesizer = GaussianCopulaSynthesizer(
    sdv_metadata, # required
    enforce_min_max_values=True,
    enforce_rounding=True,
    default_distribution='norm'
    )
    
    synthesizer.fit(df)
    return synthesizer

