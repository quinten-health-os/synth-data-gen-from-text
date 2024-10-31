import pandas as pd
import logging
import random

def categorize_columns(df: pd.DataFrame, threshold: int = 5) -> dict:
    """Categorizing columns from a dataset as discrete or continuous according to the number of modes in it.

    Args:
        df (pd.DataFrame): Dataframe
        threshold (int, optional): Number of maximum modes for a feature ti be considered discrete. Defaults to 5.

    Returns:
        dict: Keys are the category of column, values are their names. One key can have several values.
    """
    discrete_cols = []
    continuous_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if (
                len(df[col].unique()) <= threshold
            ):  # Adjust the threshold for discrete vs. continuous as needed
                discrete_cols.append(col)
            else:
                continuous_cols.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]):
            discrete_cols.append(col)

    return {"discrete": discrete_cols, "continuous": continuous_cols}

def shuffle_dict(d: dict) -> dict:
    """Shuffle a dictionary.

    Args:
        d (dict): Dictionary to shuffle

    Returns:
        dict: Shuffled dictionary
    """
    keys = list(d.keys())
    random.shuffle(keys)
    return {key: d[key] for key in keys}

def add_primary_key(df: pd.DataFrame,
                    primary_key: str) -> pd.DataFrame:
    """Add a primary key to a dataframe if not present
    Args:
        df (pd.DataFrame): Dataframe
        primary_key (str): Name of the primary key

    Returns:
        pd.DataFrame: Dataframe with a primary key
    """
    if primary_key not in df.columns:
        df[primary_key] = range(df.shape[0])
    else:
        # primary key is not unique which can happen when geenrating synthetic data
        if len(df[primary_key].drop_duplicates()) < df.shape[0]:
            logging.info("Primary key is not unique, generating a new one")
            df[primary_key] = range(df.shape[0])    
    return df

def rm_null_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values from a dataframe
    Reindexed the dataframe after removing rows
    Args:
        df (pd.DataFrame): Dataframe
    Returns:
        pd.DataFrame: Dataframe without missing values
    """
    df = df[~df.isnull().any(axis=1)] \
                .reset_index() \
                .drop(columns="index")
    return df