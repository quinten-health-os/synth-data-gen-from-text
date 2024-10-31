
import pandas as pd
import numpy as np
from typing import List

def main_ppmi_preparing(df_ppmi: pd.DataFrame,
                        columns: List,
                        drop_na: bool=False):
    """Prepare ppmi dataframe 

    Args:
        df_ppmi (DataFrame): PPMI dataframe
        columns (list): columns to keep
        drop_na (bool, optional): Drop rows with null. Defaults to False.

    Returns:
        _type_: _description_
    """
    df_ppmi = df_ppmi[columns]
    # baseline events
    df_ppmi = df_ppmi[df_ppmi['EVENT_ID'] == 'BL']
    df_ppmi = df_ppmi.drop('EVENT_ID', axis=1)
    
    # APPRDX == 1
    df_ppmi = df_ppmi[df_ppmi['APPRDX'] == 1]
    df_ppmi = df_ppmi.drop('APPRDX', axis=1)
    
    if drop_na: 
        df_ppmi = df_ppmi.dropna(axis=0)
    return df_ppmi

def main_ppmi2024_preparing(df_ppmi: pd.DataFrame,
                            columns: List,
                            drop_na: bool=False):
    """Prepare ppmi dataframe 

    Args:
        df_ppmi (DataFrame): PPMI dataframe
        columns (list): columns to keep
        drop_na (bool, optional): Drop rows with null. Defaults to False.

    Returns:
        _type_: _description_
    """
    df_ppmi = df_ppmi[columns]
    
    # baseline events
    df_ppmi = df_ppmi[df_ppmi['EVENT_ID'] == 'BL']
    df_ppmi = df_ppmi.drop('EVENT_ID', axis=1)
    
    # Parkinson patients
    df_ppmi = df_ppmi[df_ppmi['COHORT'] == 1]
    df_ppmi = df_ppmi.drop('COHORT', axis=1)
    
    if drop_na: 
        df_ppmi = df_ppmi.dropna(axis=0)
     
    return df_ppmi




def main_adni_preparing(df: pd.DataFrame,
                        columns: List,
                        drop_na: bool=False):
    """Prepare ppmi dataframe 

    Args:
        df_ppmi (DataFrame): PPMI dataframe
        columns (list): columns to keep
        drop_na (bool, optional): Drop rows with null. Defaults to False.

    Returns:
        _type_: _description_
    """
    df = df[columns]
    # baseline events
    df = df[df['VISCODE'] == 'bl']
    df = df.drop('VISCODE', axis=1)
    
    # AD patients
    df = df[df['DX_bl'] == 'AD']
    df = df.drop('DX_bl', axis=1)
    
    # remapping of PTGENDER column (for practicity of providing results)
    df = df.replace({'PTGENDER': {'Male':0, 'Female': 1}})
        
    if drop_na: 
        df = df.dropna(axis=0)
        
    # mapping columns to correct type
    for col in ['Ventricles_bl', 'ICV_bl']:
        df[col] = df[col].astype(float)
   
    return df