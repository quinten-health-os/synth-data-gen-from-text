import sys
import os
import logging

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from sklearn.model_selection import train_test_split
from src.parsers.pipeline_parser import pipeline_parser
from src.logger import init_logger
from src import loading
from src.utils import utils_sdv
from src.preparing.ppmi.preparing import main_adni_preparing
import config as conf


def main():
    
    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----Preparing PPMI data-----")

    # loading and preprocessing
    df_adni = loading.read_data(conf.BUCKET_NAME, conf.PATH_RAW_DATA, conf.FILE_PPMI_RAW_DATA)
    df_adni_prepared = main_adni_preparing(df=df_adni, 
                                            columns=conf.LIST_FTR,
                                            drop_na=True)
    # metadata dictionary
    logging.info("Creating metadata dictionary")
    metadata = utils_sdv.get_metadata_from_df(df=df_adni_prepared)
    metadata.update_column(column_name='ICV_bl', sdtype='numerical')
    metadata.update_column(column_name='Ventricles_bl', sdtype='numerical')
    metadata.update_column(column_name='MMSE', sdtype='numerical')
    logging.info(metadata)
    utils_sdv.check_metadata(metadata=metadata,
                            primary_key=conf.COL_PTID)
    
    # option of creating a training and testing set
    train_test_splits = conf.train_test_splits 
    if train_test_splits:
        logging.info(f"Creating training and testing sets")
        for k, dict_split in train_test_splits.items():
            X_train, X_test = train_test_split(df_adni_prepared,
                                               test_size=dict_split.get("split"),
                                               random_state=dict_split.get("random_state"))
            if args.save:
                loading.save_csv(X_train,
                                 conf.BUCKET_NAME,
                                 os.path.join(conf.PATH_PREPARED_DATA, "train_test_splits"),
                                 f"X_train_{k}.csv")

                loading.save_csv(X_test,
                                 conf.BUCKET_NAME,
                                 os.path.join(conf.PATH_PREPARED_DATA, "train_test_splits"),
                                 f"X_test_{k}.csv")
    # saving
    if args.save:
        loading.save_csv(df_adni_prepared,
                         conf.BUCKET_NAME,
                         conf.PATH_PREPARED_DATA,
                         conf.FILE_PREPARED_DATA)
        loading.save_dict(metadata.to_dict(),
                          conf.BUCKET_NAME,
                          conf.PATH_METADATA,
                          conf.FILE_METADATA)


if __name__ == "__main__":

    main()
