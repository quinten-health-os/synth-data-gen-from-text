import pandas as pd
import matplotlib.pyplot as plt
from sdv.single_table  import TVAESynthesizer

def fit_tvae(df: pd.DataFrame,
             sdv_metadata: dict,
             epochs: int=300,
             batch_size: int=500):
    """Train TVAE model, SDV default values"""
    # init metadata
    synthesizer = TVAESynthesizer(sdv_metadata,
                                enforce_min_max_values=True,
                                enforce_rounding=True,
                                epochs=epochs,
                                batch_size=batch_size, cuda=True)
    
    synthesizer.fit(df)
    return synthesizer


def get_loss_tvae(synthesizer: TVAESynthesizer):
    """Create a loss plot of training of a synthesizer"""
    fig = plt.figure(figsize=(10, 5)) 
    df_loss =  synthesizer.get_loss_values()
    # gavg loss by epoch
    df_loss = df_loss[["Epoch", "Loss"]].groupby("Epoch").mean().reset_index()
    epochs = df_loss.Epoch.values
    loss = df_loss.Loss.values
    plt.plot(epochs, loss, label="SDG model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    fig = plt.gcf()
    return fig
    
    