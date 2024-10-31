import subprocess
import logging
import os

def run_script(name_script: str,
               dict_args):
    """ Runs a script with arguments
    Args:
        name_script: str: name of the script to run
        **kwargs: dict: arguments to pass to the script
    """
    command = ['python3', name_script]
    for key, value in dict_args.items():
        # replace to match argparse format
        key = key.replace('_', '-') 
        command.append(f'--{key}={str(value)}')
    result = subprocess.run(command, capture_output=True, text=True)
    logging.info(result.stderr)
    return None

def get_real_train_dataset_path(path_folder: str, k: str):
    """Returns path to real training dataset of split k"""
    return os.path.join(path_folder,
                        "train_test_splits",
                        f"X_train_{k}.csv")
    
def get_real_test_dataset_path(path_folder: str, k: str):
    """Returns path to real test dataset of split k"""
    return os.path.join(path_folder,
                        "train_test_splits",
                        f"X_test_{k}.csv")
    
def get_synth_dataset_path(path_folder: str, k: str, r: int):
    """Returns path to synthetic dataset trained on real training
    dataset of split k - number of run number r"""
    return os.path.join(path_folder,
                        "train_test_splits",
                        f"X_synth_{k}_run{r}.csv")