import os
import sys
import warnings
import logging
import mlflow

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_mlflow_parser
import config as conf
from src.utils.utils_run import run_script

def log_exp_params_to_mlflow(conf: dict):
    # DATA
    mlflow.log_param("database", conf.DATABASE)
    mlflow.log_param("real_data_filename", conf.FILE_PREPARED_DATA)

    # MODEL
    mlflow.log_param("sdg_model", conf.SDG_MODEL)
    mlflow.log_param("prompt_id", conf.PROMPT_ID)


def main():
    
     # Initiate parser
    parser = pipeline_mlflow_parser()
    args = parser.parse_args()
    
    # intialize mlflow
    mlflow.set_tracking_uri(uri=conf.MLFLOW_URI)
    
    # experiment name
    mlflow.set_experiment(experiment_name="running time")
    
    # converting list of args to a dictionary
    dict_args = vars(args)
    
    # remove mlflow args from dict_args
    del dict_args["run_name"]
    del dict_args["exp_name"]
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    
    with mlflow.start_run(run_name=f"{conf.SDG_MODEL}"):
        # Log params
        log_exp_params_to_mlflow(conf=conf)
        logging.info("-----Main pipeline-----")
        run_script(name_script="scripts/main_pipeline_train_test.py", dict_args=dict_args)
    mlflow.end_run()


if __name__ == "__main__":

    main()