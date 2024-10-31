import sys
import os
import logging

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.parsers.pipeline_parser import pipeline_parser
from src.logger import init_logger
from src import loading
from src.utils import utils_sdv
from src.preparing.ppmi.preparing import main_ppmi_preparing
import config as conf


def main():
    
    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----Preparing PPMI data-----")

    # loading and preprocessing
    df_ppmi = loading.read_data(conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf.FILE_PPMI_RAW_DATA)
    df_ppmi_prepared = main_ppmi_preparing(df_ppmi=df_ppmi, 
                                            columns=conf.LIST_FTR,
                                            drop_na=True)
    # metadata dictionary
    logging.info("Creating metadata dictionary")
    metadata = utils_sdv.get_metadata_from_df(df=df_ppmi_prepared)
    utils_sdv.check_metadata(metadata=metadata,
                            primary_key=conf.COL_PTID)
    
    if args.save:
        # saving
        loading.save_csv(df_ppmi_prepared, conf.BUCKET_NAME, conf.PATH_PREPARED_DATA, conf.FILE_PREPARED_DATA)
        loading.save_dict(
            metadata.to_dict(), conf.BUCKET_NAME, conf.PATH_METADATA, conf.FILE_METADATA
        )


if __name__ == "__main__":

    main()
