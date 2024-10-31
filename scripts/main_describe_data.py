import os
import sys
import warnings

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import logging
import pandas as pd
import numpy as np

import config as conf
from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
from src.loading import read_data, read_dict
from src.loading import save_figure_s3, save_csv
from src.utils.utils_sdv import get_metadata_from_dict
from src.utils.utils_df import categorize_columns
from src.evaluating.descriptive_statistics import main_cat_stats_descs
from src.evaluating.descriptive_statistics import main_num_stats_descs
from src.visualizing.vis_sdv import get_distrib_and_corr_plots


def main():
    
    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----Description of data-----")

    df_real = read_data(conf.BUCKET_NAME, conf.PATH_PREPARED_DATA, conf.FILE_PREPARED_DATA)
    df_synth = read_data(conf.BUCKET_NAME, conf.PATH_SYNTH_DATA, conf.FILE_SYNTHESIZED_DATA)

    dict_metadata = read_dict(
            conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, conf.FILE_METADATA)
        )

    sdv_metadata = get_metadata_from_dict(dict_metadata)
        
    # Get num & cat columns
    dict_col = categorize_columns(df_real)
    list_var_cat = dict_col.get("discrete")
    list_var_num = dict_col.get("continuous")
    if conf.COL_PTID in list_var_num:
        list_var_num.remove(conf.COL_PTID)
    
    # viz distribs
    fig_cache_distrib, fig_cache_corr = get_distrib_and_corr_plots(
        df_real=df_real,
        df_synth=df_synth,
        meta_data=sdv_metadata,
        list_col_num=list_var_num)

    # compute descriptive stats
    df_real["type"] = "real"
    df_synth["type"] = "synth"
    
    # in case ptid column not in original / synthetic dataframes add mock ones
    if conf.COL_PTID not in df_real:
        df_real[conf.COL_PTID] = np.arange(df_real.shape[0])
    if conf.COL_PTID not in df_synth:
        df_synth[conf.COL_PTID] = np.arange(df_synth.shape[0])

    df = pd.concat([df_real, df_synth], axis=0)

    # categorical stat descs
    logging.info("Compute descriptive stats")
    df_cat_stats_descs = main_cat_stats_descs(df=df,
                                            list_var_cat=list_var_cat,
                                            list_col_aggr=['type'],
                                            col_id=conf.COL_PTID)
    # numerical stat descs
    df_num_stats_descs = main_num_stats_descs(df=df,
                                            list_var_num=list_var_num,
                                            list_col_aggr=["type"],
                                            col_id=conf.COL_PTID)
    if args.save:
        # save dataframes and plots
        save_csv(df_cat_stats_descs,
                conf.BUCKET_NAME,
                os.path.join(conf.PATH_EVALUATE, f"stat_descs/{conf.DATABASE}"),
                f"df_cat_stats_descs_{conf.DATE}.csv")
        
        save_csv(df_num_stats_descs,
                conf.BUCKET_NAME,
                os.path.join(conf.PATH_EVALUATE, f"stat_descs/{conf.DATABASE}"),
                f"df_num_stats_descs_{conf.DATE}.csv")

        for name, fig in fig_cache_distrib.items():
            save_figure_s3(
                fig,
                conf.BUCKET_NAME,
                conf.PATH_EVALUATE,
                f"viz_data/{conf.DATABASE}/distrib/{name}",
            )
        
        for name, fig in fig_cache_corr.items():
            save_figure_s3(
                fig,
                conf.BUCKET_NAME,
                conf.PATH_EVALUATE,
                f"viz_data/{conf.DATABASE}/correlation/{name}",
            )


if __name__ == '__main__':

    main()