import pandas as pd
from sdv.single_table  import CTGANSynthesizer

def fit_ctgan(df: pd.DataFrame,
             sdv_metadata: dict,
             epochs: int=300,
             batch_size: int=500):
    """Train CTGAN model, SDV default values"""
     # init metadata
    synthesizer = CTGANSynthesizer(sdv_metadata,
                                enforce_min_max_values=True,
                                enforce_rounding=True,
                                epochs=epochs,
                                batch_size=batch_size, cuda=True,
                                verbose=True)
    
    synthesizer.fit(df)
    return synthesizer




    