"""Functions to support descriptive statistics"""

import pandas as pd
from typing import List

from src.utils.utils_stat import q1, q3, set_to_str
from src.utils.utils_stat import flatten_multiindex, replace_na_in_col

def get_stats_descs(df: pd.DataFrame,
                    var_type: str,
                    list_var: List[str],
                    list_col_aggr: List[str],
                    list_agg_func: list,
                    col_id: str) -> pd. DataFrame:
    # quality check on listed variables to compute stats for
    list_var_not_in_df =  [var for var in list_var if var not in df.columns]
    assert len(list_var_not_in_df) == 0, f"Following variables in list of variables not present in dataframe: {list_var_not_in_df}"

    # pass columns to aggreate on into str
    # for simplicity when creating columns from pivot multiindex
    df = set_to_str(df=df, list_cols=list_col_aggr)

    # create table with 1 row / PTID | Variable | Category
    df_melt = pd.melt(df, id_vars=list_col_aggr + [col_id], value_vars=list_var, var_name="Variable", value_name="Category")

    # compute stats with pivot function 
    if var_type == 'categorical':
        index = ["Variable", "Category"]
    else:
        index = ["Variable"]
        # remove unuseful columns in numerical case
        # in numerical case pivot does not support additional cols
        df_melt = df_melt.drop(columns=[col_id])
    df_stats = pd.pivot_table(df_melt,
                              index=index,
                              columns=list_col_aggr,
                              aggfunc=list_agg_func)

    # flatten columns' mutiindex to get 1 dimensional column index
    df_stats = df_stats.droplevel(1, axis=1)
    df_stats.columns = flatten_multiindex(col_names=df_stats.columns)

    # get name of aggregation groups, useful to compute derivative columns over aggregation groups
    aggr_groups = [col.replace("COUNT_", "") for col in df_stats.columns]
    df_stats = df_stats.reset_index()

    # replace na by 0 in count columns
    list_col_count = df_stats.filter(regex="^COUNT")
    df_stats = replace_na_in_col(df=df_stats, list_col=list_col_count, value=0)
 
    if var_type == 'categorical': 
        # compute total count by category
        df_count_tot = df_stats.drop(columns=["Category"]).groupby("Variable").sum()
        
        # formatting: replace count by tot in count tot dataframe
        df_count_tot.columns = [col.replace("COUNT", "TOT") for col in df_count_tot.columns]
            
        # add tot count information 
        df_stats = df_stats.merge(df_count_tot, on="Variable", how="inner")
            
        # compute prevalence 
        for col in aggr_groups:
            df_stats[f"PCT_{col}"] = df_stats[f"COUNT_{col}"]/ df_stats[f"TOT_{col}"]

    return df_stats
  

def main_cat_stats_descs(df: pd.DataFrame, 
                        list_var_cat: List[str],
                        list_col_aggr: List[str],  
                        col_id: str) -> pd.DataFrame:
    """Retrieves categorical stat descs from a feature matrix given a list of variables and modalities to aggregate stats on.
    This function contains a list_agg_func argument which is the list of functions to compute : it has to be equal to ['count'] because the categorical stat descs are computed.
    NB: This function deals with null values. 

    Args:
        df (pd.DataFrame): 1 row / PTID | variable_name 
        list_var_cat (list): list of variables to compute stats on
        list_col_aggr (list): list of modalities to aggregate stats on
        col_id (str): name of patient id column

    Returns:
        pd. DataFrame: 1 col / variable | (category) & 1 col / function & modality

    Example:
    Inputs:
        df: 1 row / PTID | gender (F or M)| outcome (0 or 1)
        list_var_cat = ["gender"]
        list_col_aggr = ["label"]
        col_id = "PTID"
    Output:
        df: 1 row / gender | count_0 | count_1 | tot_0 | tot_1 | pct_0 | pct_1
    """
    df = df.fillna("Missing value")

    # compute stats for entire dataframe without any aggregation
    df = get_stats_descs(df=df,
                var_type='categorical',
                list_var=list_var_cat,
                list_col_aggr=list_col_aggr,
                list_agg_func=["count"],
                col_id=col_id)

    df = df.query("Category == 1")
    
    return df

def main_num_stats_descs(df: pd.DataFrame, 
                   list_var_num: List[str],
                   list_col_aggr: List[str],  
                   col_id: str) -> pd.DataFrame:
    """Retrieves numerical stat descs from a feature matrix given a list of variables and modalities to aggregate stats on.
    This function contains a list_agg_func argument which is the list of functions to compute :
        Accepted : list containing any of these values : 'count', 'mean', 'std', 'min', 'median', 'max', q1, q3
        Default : equals to ['count', 'mean', 'std', 'min', 'median', 'max', q1, q3]
    NB: This function deals with null values. 
    
    Args:
        df (pd.DataFrame): 1 row / PTID | variable_name 
        list_var_num (list): list of variables to compute stats on
        list_col_aggr (list): list of modalities to aggregate stats on
        col_id (str): name of patient id column

    Returns:
        pd. DataFrame: 1 col / variable | (category) & 1 col / function & modality
        
    Example:
    Inputs:
        df: 1 row / PTID | age | outcome (0 or 1)
        list_var_num = ["age"]
        list_col_aggr = ["label"]
        col_id = "PTID"
    Output:
        df: 1 row / age | mean_0 | mean_1 |std_0 | std_1 
    """
    df = get_stats_descs(df=df,
                var_type='numerical',
                list_var=list_var_num,
                list_col_aggr=list_col_aggr,
                list_agg_func=['mean', 'std', 'min', q1, 'median', q3, 'max', 'count'],
                col_id=col_id)
    
    return df 

