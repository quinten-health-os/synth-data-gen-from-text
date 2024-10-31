import os
import sys
import warnings
import logging
import pandas as pd

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config as conf
from src import loading
from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
from src.evaluating import evaluate_fidelity
from src.utils import utils_sdv


def main():
    
    # initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()

    # initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----SDG fidelity evaluation-----")

    # loading data
    df_real = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_PREPARED_DATA, conf.FILE_PREPARED_DATA
    )
    df_synth = loading.read_data(
        conf.BUCKET_NAME, conf.PATH_SYNTH_DATA, conf.FILE_SYNTHESIZED_DATA
    )
    dict_metadata = loading.read_dict(
        conf.BUCKET_NAME, os.path.join(conf.PATH_METADATA, conf.FILE_METADATA)
    )
    sdv_metadata = utils_sdv.get_metadata_from_dict(dict_metadata=dict_metadata)
    logging.info("Data loaded")
    
    # check validity of data with metadata dict and 
    # transform dataframe if necessary (e.g. add primary key column, correct data type)
    df_real = utils_sdv.custom_validate_data(df=df_real, metadata=sdv_metadata)
    df_synth = utils_sdv.custom_validate_data(df=df_synth, metadata=sdv_metadata)
    
    sdv_report = utils_sdv.get_sdv_report(df_real, df_synth, sdv_metadata)
    df_base_metrics = sdv_report.get_properties()
    df_base_metrics = df_base_metrics.rename(columns={"Property": "Metric", "Score": "Value"})

    # get sdv plots
    corr_plot = evaluate_fidelity.get_correlation_plot(sdv_report)
    score_plot = evaluate_fidelity.get_score_plot(sdv_report)

    # compute metrics
    dict_metrics = evaluate_fidelity.evaluate_fidelity(df_real=df_real,
                                                       df_synth=df_synth,
                                                       metadata=sdv_metadata,
                                                       list_metrics=conf.FIDELITY_METRICS_TO_COMPUTE)
    df_metrics = pd.DataFrame(list(dict_metrics.items()), columns=['Metric', 'Value'])
    df_metrics = pd.concat([df_base_metrics, df_metrics], axis=0)
    
    logging.info("-------------Fidelity metrics-------------")
    logging.info(df_metrics)
    
    if args.save:
        
        loading.save_csv(
            df_metrics,
            conf.BUCKET_NAME,
            os.path.join(conf.PATH_EVALUATE, f"evaluate_fidelity/{conf.DATABASE}", "dataframes/"),
            f"df_metrics_fidelity_{conf.DATABASE}_{conf.DATE}.csv"
        )
        
        # saving plots
        loading.save_figure_s3(
        corr_plot,
        conf.BUCKET_NAME,
        os.path.join(conf.PATH_EVALUATE, f"evaluate_fidelity/{conf.DATABASE}", "plots/"),
        f"corr_plot_fidelity_{conf.DATABASE}_{conf.DATE}.csv",
        )

        loading.save_figure_s3(
            score_plot,
            conf.BUCKET_NAME,
            os.path.join(conf.PATH_EVALUATE, f"evaluate_fidelity/{conf.DATABASE}", "plots/"),
            f"score_plot_fidelity_{conf.DATABASE}_{conf.DATE}",
        )

if __name__ == "__main__":

    main()
