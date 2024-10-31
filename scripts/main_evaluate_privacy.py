import os
import sys
import warnings
import logging
import pandas as pd

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
import config as conf
from src import loading
from src.utils import utils_sdv
from src.evaluating import evaluate_privacy

def main():
   
    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----SDG privacy evaluation-----")
    
    # Loading data
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
    
    # check validity of data with metadata dict
    df_real = utils_sdv.custom_validate_data(df=df_real, metadata=sdv_metadata)
    df_synth = utils_sdv.custom_validate_data(df=df_synth, metadata=sdv_metadata)

    # Evaluate privacy
    dict_metrics = evaluate_privacy.evaluate_privacy(df_real=df_real,
                                                     df_synth=df_synth,
                                                     metadata=sdv_metadata,
                                                     list_metrics=conf.PRIVACY_METRICS_TO_COMPUTE,
                                                     sensitive_fields=conf.LIST_SENSITIVE_FIELDS,
                                                     key_fields=conf.LIST_KEY_FIELDS,
                                                     sample_size=None) # use 100% of the synthetic data to compute metrics 
    df_metrics = pd.DataFrame(list(dict_metrics.items()), columns=['Metric', 'Value'])
    
    logging.info("-------------Privacy metrics-------------")
    logging.info(df_metrics)
    
    if args.save:

        # saving dataframes
        loading.save_csv(
            df_metrics,
            conf.BUCKET_NAME,
            os.path.join(conf.PATH_EVALUATE, f"evaluate_privacy/{conf.DATABASE}"),
            f"df_metrics_privacy_{conf.DATABASE}_{conf.DATE}.csv"
        )


if __name__ == "__main__":

    main()
