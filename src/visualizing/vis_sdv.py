import pandas as pd
import cachetools
import logging
from sdv.evaluation.single_table import get_column_plot, get_column_pair_plot


def get_distrib_and_corr_plots(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    meta_data: dict,
    list_col_num: list,
):
    """
    Get the graphs of all distribution plots (real vs synthetic data) of sdv report and store them in a cache that is returned.
    Get the graphs of all correlation plots (real vs synthetic data) of sdv report and store them in a cache that is returned.
    Returns a tuple with both caches.
    """
    nb_fig_1 = len(df_real.columns)
    nb_fig_2 = len(list_col_num) ** 2
    figure_cache_column_plot = cachetools.LRUCache(maxsize=nb_fig_1)
    figure_cache_column_pair_plot = cachetools.LRUCache(maxsize=nb_fig_2)

    for col in df_real.columns:
        if col != meta_data.primary_key:        
            fig = get_column_plot(
                real_data=df_real,
                synthetic_data=df_synth,
                column_name=col,
                metadata=meta_data,
            )
            figure_cache_column_plot[col] = fig
    logging.info("Create all distribution plots")
    
    for i in range(len(list_col_num)):
        for j in range(i + 1, len(list_col_num)):
            fig = get_column_pair_plot(
                real_data=df_real,
                synthetic_data=df_synth,
                column_names=[list_col_num[i], list_col_num[j]],
                metadata=meta_data,
            )
            figure_cache_column_pair_plot[
                f"{list_col_num[i]}_{ list_col_num[j]}"
            ] = fig
            
    logging.info("Create all correlation plots")

    return figure_cache_column_plot, figure_cache_column_pair_plot