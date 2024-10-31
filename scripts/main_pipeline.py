import os
import sys
import warnings


warnings.filterwarnings("ignore")
script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

from src.logger import init_logger
from src.parsers.pipeline_parser import pipeline_parser
import config as conf
from src.utils.utils_run import run_script

def main():
    
     # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # converting list of args to a dictionary
    dict_args = vars(args)
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
   
    pipeline_steps = conf.PIPELINE_STEPS_TO_PERFORM
    
    for step in pipeline_steps:
        if step == 'preparing':
            name_script = f'scripts/main_preparing_{conf.DATABASE}.py'
        else:
            name_script = f'scripts/main_{step}.py'
        run_script(name_script=name_script, dict_args=dict_args)

if __name__ == "__main__":

    main()