import os
import sys
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
from src.utils.utils_run import run_script
from src.utils.utils_run import get_real_train_dataset_path
from src.utils.utils_run import get_real_test_dataset_path
from src.utils.utils_run import get_synth_dataset_path
import config as conf


def main():
    
     # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # converting list of args to a dictionary
    dict_args = vars(args)
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
   
    pipeline_steps = conf.PIPELINE_STEPS_TO_PERFORM
    train_test_splits = conf.train_test_splits
    
    total_steps = sum(len(split.get("n_runs")) for split in train_test_splits.values())
    pbar = tqdm(total=total_steps, desc="Pipeline Progress", unit="step")
    
    for step in pipeline_steps:
        if step == 'preparing':
            name_script = f'scripts/main_preparing_{conf.DATABASE}.py'
            run_script(name_script=name_script, dict_args=dict_args)
            
        elif step == "evaluate_agg":
            name_script = f'scripts/main_{step}.py'
            run_script(name_script=name_script, dict_args=dict_args)
   
        else:
            name_script = f'scripts/main_{step}.py'
            
            # run throught each train/ test split 
            for k, dict_split in train_test_splits.items():
                
                # run throught each run of the synthetic data generation process
                for r in dict_split.get("n_runs"):
                    
                    # get training path name
                    dict_args["real_dataset"] = get_real_train_dataset_path(k=k,
                                                                            path_folder=conf.PATH_PREPARED_DATA)
                    
                    # get synthetic path name
                    # dict_args["synth_dataset"] = get_synth_dataset_path(k=k, r=r,
                    #                                                     path_folder=conf.PATH_SYNTH_DATA)
                    
                    dict_args["synth_dataset"] =  get_real_test_dataset_path(k=k,
                                                                            path_folder=conf.PATH_PREPARED_DATA)
                    
                    
                    dict_args["test_dataset"] = get_real_test_dataset_path(k=k,
                                                                            path_folder=conf.PATH_PREPARED_DATA)
                    
                    # run script
                    run_script(name_script=name_script, dict_args=dict_args)
                    
                    if step == "tab_to_tab_sdg":
                        # update pbar only when synthetic data is generated (longest step)
                        pbar.update(1)    
                        
                    elif step == "text_to_tab_sdg_shuffle":
                        # update pbar only when synthetic data is generated (longest step)
                        pbar.update(1)  
                        
                    # evaluation: rerun with real dataset = X_test
                    elif step == "evaluate":
                        # change real dataset in args to test dataset
                       dict_args["real_dataset"] = get_real_test_dataset_path(k=k,
                                                                              path_folder=conf.PATH_PREPARED_DATA) 
                       run_script(name_script=name_script, dict_args=dict_args)
                       
                    else:
                        pass
  
    pbar.close()
                
            
if __name__ == "__main__":

    main()