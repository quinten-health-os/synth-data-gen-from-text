import sys
import os
import logging

script_dir = os.path.dirname(os.path.abspath("src/"))
sys.path.append(script_dir)

import config as conf
from src.parsers.pipeline_parser import pipeline_parser
from src.logger import init_logger
from src.prompt_engineering.prompt_text_to_tab import prompt_synth_tab
from src.prompt_engineering.utils_prompt import parse_prompt
from src.loading import save_csv, save_text


def main():
    
    # Initiate parser
    parser = pipeline_parser()
    args = parser.parse_args()
    
    # Initiate logger
    init_logger(level=args.log_level, file=True, file_path="logs/logs.txt")
    logging.info("-----Text to tabular SDG-----")
    logging.info(f"Model: {conf.SDG_MODEL}")
    logging.info(f"Size of synthetic database: {conf.N_SAMPLE}")
    logging.info(f"Prompt ID: {conf.PROMPT_ID}")
    
    prompt = parse_prompt(prompt_dict=conf.TEXT2TAB_PROMPT_DICT[conf.PROMPT_ID],
                            prompt_example=conf.ROW_EXAMPLE,
                            var_desc_prompt_dict=conf.VAR_DESC_PROMPT_DICT,
                            ref_key=conf.REFERENTIAL_VAR_NAME,
                            shuffle=True)

    df_synth = prompt_synth_tab(api_key=conf.OPENAPI_KEY,
                     prompt=prompt,
                     model=conf.SDG_MODEL,
                     n_rows=conf.N_ROWS,
                     n_sample=conf.N_SAMPLE,)
    if args.save:
        # saving data
        save_csv(df_synth,
                conf.BUCKET_NAME,
                conf.PATH_SYNTH_DATA, 
                conf.FILE_SYNTHESIZED_DATA,
                )
        save_text(prompt,
                conf.BUCKET_NAME,
                conf.PATH_SYNTH_DATA, 
                conf.FILE_SYNTHESIZED_DATA_PROMPT,
                )
if __name__ == "__main__":
    
    main()