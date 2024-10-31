import logging
import pandas as pd
from tqdm import tqdm

from src.prompt_engineering.prompt_llm import prompt_model
from src.prompt_engineering.prompt_llm import extract_json_as_dict
   
def prompt_synth_tab(prompt: str,
                     model: str,
                     n_rows: int,
                     n_sample: int,
                     role: str="user",
                     show_progress: bool=True) -> str:
    """
    Generates a synthetic tabular dataframe from a text describin the
    dataset to generate.
    The prompt must include the list of columns and their respective type to include.
    If ambiguous, columns shuold be described or units provided.
    Args:
        api_key (str): OpenAI API key
        prompt (str): text to generate the synthetic dataset
        model (strà): OpenAI model to use
        n_rows (int): number of rows to generate for each request to the API
        n_sample (int): number of samples to generate
        role (str): role of the user
    """
    synth_data = []
    n_iter = n_sample // n_rows
    
    if show_progress:
        pbar = tqdm(total=n_sample, desc="Synth data queries")
    k = 0
    
    while k < n_iter:
        
        logging.info(f"Synth data query n°{k}")
        msg = prompt_model(model=model,
                            prompt=prompt,
                            role=role)
        dictionary = extract_json_as_dict(msg)
        
        if dictionary:
            df = pd.DataFrame.from_dict(dictionary, orient='index')
            synth_data.append(df)
            if show_progress:
                pbar.update(len(df))
            k += 1
        else:
            logging.info("No dictionary")
            
    if show_progress:
        pbar.close()
        
    # formatting synthetic data into a dataframe
    df_synth = pd.concat(synth_data, axis=0)
    logging.info(f"Shape of synthetic dataframe: {df_synth.shape}")
    
    return df_synth
            