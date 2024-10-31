import os
import sys
import warnings
import logging
import mlflow
import pandas as pd
import mlflow

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config as conf
from src import loading
from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser

def log_exp_params_to_mlflow(conf: dict):
    # DATA
    mlflow.log_param("database", conf.DATABASE)
    mlflow.log_param("real_data_filename", conf.FILE_PREPARED_DATA)

    # MODEL
    mlflow.log_param("sdg_model", conf.SDG_MODEL)
    mlflow.log_param("prompt_id", conf.PROMPT_ID)

def main():
    # initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()

    # initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----SDG fidelity evaluation-----")

    train_test_splits = conf.train_test_splits
    list_df_metrics = []
    
    # run throught each train/ test split 
    for k, dict_split in train_test_splits.items():
        # run throught each run of the synthetic data generation process
        for r in dict_split.get("n_runs"):
            
            # load df metrics x_synth vs. X_train
            for split in ["train", "test"]:
                filename = f"df_metrics_X_{split}_{k}_vs_X_synth_{k}_run{r}.csv"
                path_file = os.path.join(conf.PATH_EVALUATE, filename)
                df_metrics = loading.read_data(conf.BUCKET_NAME, path_file)
                df_metrics["split"] = split
                df_metrics["split_n"] = k
                df_metrics["run"] = r
                list_df_metrics.append(df_metrics)
        
    df_metrics_all = pd.concat(list_df_metrics, axis=0)
    df_metrics_all_ = df_metrics_all.drop(columns=["split_n", "run"])
    
    # compute mean / MCSE for each metric
    df_metrics_agg = pd.pivot_table(df_metrics_all_,
                                  index=["Metric"],
                                  values= "Value",
                                  columns=["split"],
                                  aggfunc=["mean", "std"])
    # flatten, the columns

    df_metrics_agg.columns = ['_'.join(col).strip() for col in df_metrics_agg.columns.values]
    
    # reorder columns in desired order
    df_metrics_agg = df_metrics_agg[['mean_train', 'std_train', 'mean_test', 'std_test']]
    
    # reorder index
    df_metrics_agg = df_metrics_agg.reindex(["Size real", "Size synth", "Column Shapes", "Column Pair Trends"] + conf.FIDELITY_METRICS_TO_COMPUTE + conf.PRIVACY_METRICS_TO_COMPUTE + conf.UTILITY_METRICS_TO_COMPUTE + ["BinaryAdaBoostClassifier_aug"])
    
    # Reset the index to have 'Metrics' as a column
    logging.info("-------------SDG metrics aggregation-------------")
    df_metrics_agg = df_metrics_agg.reset_index().rename(columns={'index': 'Metrics'})
    
    if args.save:
        
        loading.save_csv(
            df_metrics_agg,
            conf.BUCKET_NAME,
            conf.PATH_EVALUATE,
            f"df_metrics_{conf.SDG_MODEL}_synth_train_test_agg.csv")
        
        loading.save_csv(
            df_metrics_all,
            conf.BUCKET_NAME,
            conf.PATH_EVALUATE,
            f"df_metrics_splits_{conf.SDG_MODEL}_synth_train_test_agg.csv")
        
    # saving to mlflow
    # intialize mlflow
    mlflow.set_tracking_uri(uri=conf.MLFLOW_URI)
    # experiment name
    mlflow.set_experiment(experiment_name="benchmark_metrics")
    
    with mlflow.start_run(run_name=f"{conf.SDG_MODEL}"):
        if "log_mlflow" in args and args.log_mlflow:
            
            for i, row in df_metrics_agg.iterrows():
                mlflow.log_metric(f"{row['Metric']}_train_mean", row["mean_train"])
                mlflow.log_metric(f"{row['Metric']}_test_mean", row["mean_test"])
                mlflow.log_metric(f"{row['Metric']}_train_sem", row["std_train"])
                mlflow.log_metric(f"{row['Metric']}_test_sem", row["std_test"])
    mlflow.end_run()
    

if __name__ == "__main__":

    main()
