import os
import json
import logging

from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

def prompt_model(model: str,
                prompt: str,
                role: str="user"):
    """Prompt the LLM model with the given prompt

    Args:
        model (str): LLM model. Either 'gpt' or 'mistral'.
        prompt (str): Prompt to be used.
        role (str, optional): user role used in prompting. Defaults to "user".

    Returns:
        message (str): Response from the model
    """
    if 'gpt' in model:
        msg = prompt_openai_model(model=model,
                        prompt=prompt,
                        role=role)
    elif 'mistral' in model:
        msg = prompt_mistral_model(model=model,
                         role=role,
                         prompt=prompt)
    else:
        logging.info("Model not recognized")
        msg = None
    return msg


def prompt_openai_model(model: str,
                        prompt: str,
                        role: str="user"):
    """Prompt the OpenAI model with the given prompt

    Args:
        model (str): OpenAI model. Either 'gpt' 
        prompt (str): Prompt to be used.
        role (str, optional): user role used in prompting. Defaults to "user".

    Returns:
        message (str): Response from the model
    """
    # add in .zschrc file "export MISTRAL_API_KEY='%yourkey'"
    # run in terminal: source ~/.zshrc	
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        logging.info("OPENAI_API_KEY not found")
        return None
    # connect to openai API via client
    client = OpenAI(api_key=api_key)

    # prompt the model
    res = client.chat.completions.create(
        messages=[{"role": role,
                   "content": prompt,
                   }],
        model=model,
    )
    
    # extract message from response
    msg = res.choices[0].message.content
    return msg


def prompt_mistral_model(model: str,
                         role: str,
                         prompt: str):
    """Prompt the MISTRAL model with the given prompt

    Args:
        model (str): mistral model
        role (str): user role used in prompting. Defaults to "user".
        prompt (str): Prompt to be used.

    Returns:
        message (str): Response from the model
    """
    # add in .zschrc file "export MISTRAL_API_KEY='%yourkey'"
    try:
        api_key = os.environ["MISTRAL_API_KEY"]
    except KeyError:
        logging.info("MISTRAL_API_KEY not found")
        return None
    # connect to mistral  API via client
    client = MistralClient(api_key=api_key)
    
    # prompt the model
    res = client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=[ChatMessage(role=role,
                              content=prompt)],
    )
    
    # extract message from response
    msg = res.choices[0].message.content
    return msg

def extract_json_as_dict(json_file: str) -> dict:
    """Extract JSON file as dictionary

    Args:
        json_file (str): JSON file

    Returns:
        dict: dictionary
    """
    try:
        dictionary = json.loads(json_file)
        return dictionary
    except(ValueError, json.JSONDecodeError):
        logging.info("JSON decode error")
        logging.info(json_file)
        return None
    