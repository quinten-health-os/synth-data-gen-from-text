import os
import sys
import warnings
import logging
import pandas as pd
import re

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config as conf
from src import loading
from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
from src.evaluating import evaluate_fidelity
from src.evaluating import evaluate_privacy
from src.evaluating import evaluate_utility
from src.utils import utils_sdv


def main():
    
    # initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()

    # initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----SDG fidelity evaluation-----")

    # loading data
    path_file_real = args.real_dataset if args.real_dataset else os.path.join(conf.PATH_PREPARED_DATA, conf.FILE_PREPARED_DATA)
    df_real = loading.read_data(conf.BUCKET_NAME, path_file_real)
    
    path_file_synth = args.synth_dataset if args.synth_dataset else os.path.join(conf.PATH_SYNTH_DATA, conf.FILE_SYNTHESIZED_DATA)
    df_synth = loading.read_data(conf.BUCKET_NAME, path_file_synth)
    
    path_file_test = args.test_dataset if args.synth_dataset else None
    df_test = loading.read_data(conf.BUCKET_NAME, path_file_test)
    
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
    
    # adding size of real and synthetic data to the metrics
    df_base_metrics.loc[2] = ["Size synth", df_synth.shape[0]]
    df_base_metrics.loc[3] = ["Size real", df_real.shape[0]]

    # get sdv plots
    corr_plot = evaluate_fidelity.get_correlation_plot(sdv_report)
    score_plot = evaluate_fidelity.get_score_plot(sdv_report)

    # compute fidelity metrics
    dict_metrics_f = evaluate_fidelity.evaluate_fidelity(df_real=df_real,
                                                       df_synth=df_synth,
                                                       metadata=sdv_metadata,
                                                       list_metrics=conf.FIDELITY_METRICS_TO_COMPUTE)
    df_metrics_f = pd.DataFrame(list(dict_metrics_f.items()), columns=['Metric', 'Value'])
    
    # compute privacy metrics
    dict_metrics_p = evaluate_privacy.evaluate_privacy(df_real=df_real,
                                                     df_synth=df_synth,
                                                     metadata=sdv_metadata,
                                                     list_metrics=conf.PRIVACY_METRICS_TO_COMPUTE,
                                                     sensitive_fields=conf.LIST_SENSITIVE_FIELDS,
                                                     key_fields=conf.LIST_KEY_FIELDS,
                                                     sample_size=None) # use 100% of the synthetic data to compute metrics 
    df_metrics_p = pd.DataFrame(list(dict_metrics_p.items()), columns=['Metric', 'Value'])
    
    # compute utility metrics
    dict_metrics_u = evaluate_utility.evaluate_utility(df_real=df_real,
                                                       df_synth=df_synth,
                                                       metadata=sdv_metadata,
                                                       col_target=conf.COL_TARGET,
                                                       list_metrics=conf.UTILITY_METRICS_TO_COMPUTE)
    df_metrics_u = pd.DataFrame(list(dict_metrics_u.items()), columns=['Metric', 'Value'])
    
    # compute utility metrics using test set
    df_synth_aug = pd.concat([df_synth, df_real], axis=0)
    cols = [col for col in df_synth_aug.columns if col in df_test.columns]
    df_test = df_test[cols]
    dict_metrics_u_aug = evaluate_utility.evaluate_utility(df_real=df_test,
                                                       df_synth=df_synth_aug,
                                                       metadata=sdv_metadata,
                                                       col_target=conf.COL_TARGET,
                                                       list_metrics=["BinaryAdaBoostClassifier"])
    dict_metrics_u_aug["BinaryAdaBoostClassifier_aug"] = dict_metrics_u_aug.pop("BinaryAdaBoostClassifier")
    df_metrics_u_aug = pd.DataFrame(list(dict_metrics_u_aug.items()), columns=['Metric', 'Value'])
    df_metrics = pd.concat([df_base_metrics, df_metrics_f, df_metrics_p, df_metrics_u, df_metrics_u_aug], axis=0)
    
    logging.info("-------------SDG metrics-------------")
    
    if args.save:
        if args.synth_dataset:
            # get names of real and synthetic data files without '.csv' 
            suffix_synth = re.search(r'([^/]+)(?=\.[^./]+$)', args.synth_dataset).group()
            suffix_real = re.search(r'([^/]+)(?=\.[^./]+$)', args.real_dataset).group()
            suffix = f"{suffix_real}_vs_{suffix_synth}"
        else:
            suffix =f'{conf.DATABASE}_{conf.DATE}'
        loading.save_csv(
            df_metrics,
            conf.BUCKET_NAME,
            conf.PATH_EVALUATE,
            f"df_metrics_{suffix}.csv"
        )
        
        # saving plots
        loading.save_figure_s3(
        corr_plot,
        conf.BUCKET_NAME,
        os.path.join(conf.PATH_EVALUATE, "plots/"),
        f"corr_plot_fidelity_{suffix}.csv",
        )

        loading.save_figure_s3(
            score_plot,
            conf.BUCKET_NAME,
            os.path.join(conf.PATH_EVALUATE, "plots/"),
            f"score_plot_fidelity_{suffix}",
        )

if __name__ == "__main__":

    main()
