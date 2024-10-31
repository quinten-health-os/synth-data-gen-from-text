import pandas as pd
import numpy as np
from typing import Union, List

def q1(x):
    """
    Retrieves first quartile
    """
    return np.nanquantile(x, 0.25)

def q3(x):
    """
    Retrieves third quartile
    """
    return np.nanquantile(x, 0.75)


def set_to_str(df: pd.DataFrame,
              list_cols: Union[str, List[str]]) -> pd.DataFrame:
    """Sets types of cols in col_list as str type

    Args:
        df (pd.DataFrame): data
        list_cols (List[str]): name of column or list of columns to change type into float

    Returns:
        pd.DataFrame: 
    """
    if not isinstance(list_cols, list):
        list_cols = [list_cols]
    for col in list_cols:
        df[col] = df[col].astype(str)
    return df


def flatten_multiindex(col_names: list,
                       preproc: bool=True) -> list:
    """Flattens multiindex into 1 dimensional index
    If preproc is true, column names are put into uppercase and ' ' replaced by '_'
   
    Args:
        col_names (list): multiindex of dimension n x m
        preproc (bool, optional): Wether to preprocess column names. Defaults to True.

    Returns:
        list: list of size n x m

    Example:
        col_names = [["Placebo", "Treatment"], ["MEAN", "STD"]]
        preproc = True
        will return 4 new column names : ["PLACEO_MEAN", "PLACEO_STD", "TREATMENT_MEAN", "TREATMENT_STD"]
    """
    col_levels = [list(col) for col in col_names]
    col_names_new = ["_".join(col) for col in col_levels]

    # preprocessing of column names
    if preproc:
        col_names_new = [col.replace(' ', '_').upper() for col in col_names_new]

    return col_names_new

def replace_na_in_col(df: pd.DataFrame,
                      list_col: list,
                      value: Union[int, str]) -> pd.DataFrame:
    
    
    mapping_col_values = {col: {np.nan: value} for col in list_col}
    df = df.replace(mapping_col_values)
    return df
    
    