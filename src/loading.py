import logging
import json
import logging
import pickle
import boto3
import os
import matplotlib
import plotly
import pickle
import pandas as pd
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg
from typing import Optional

import config as conf

def read_data(
    bucket_name: str,
    path_file: str,
    filename: str=None,
    sep: str=",",
    compression: str="infer",
) -> None:
    if filename:
        path_file = os.path.join(path_file, filename)
    path_s3 = os.path.join("s3://", bucket_name, path_file)
    logging.info("Data will be loaded from {}".format(path_file))
    df = pd.read_csv(path_s3, sep=sep, compression=compression)
    return df


def save_csv(
    df: pd.DataFrame,
    bucket_name: str,
    path_file: str,
    filename: str=None,
    index: bool=False,
) -> None:
    """
    Save a pandas dataframe into s3

    Args:
        df (DataFrame): pandas dataset to save
    bucket (str): name of the bucket (not ending with '/')
    path (str): path of the file without '/' at the beginning and ending with .csv
    Returns:
        None
    """
    if filename:
        path_file = os.path.join(path_file, filename)
    path_s3 = os.path.join("s3://", bucket_name, path_file)
    logging.info("Data will be saved in {}".format(path_s3))
    df.to_csv(path_s3, index=index)


def read_dict(bucket_name: str,
              path: str):
    """Reads a dictionary from .json file

    Args:
        bucket_name (str): name of the bucket (not ending with '/')
        path (str): path of the file without '/' at the beginning and ending with .json

    Returns:
        None
    """
    # Initialize boto3 to use S3 resource
    s3_resource = boto3.resource("s3")

    try:
        # Get the object from the S3 Bucket
        s3_object = s3_resource.Object(bucket_name=bucket_name, key=path)

        # Get the response from get_object()
        s3_response = s3_object.get()

        # Get the Body object in the S3 get_object() response
        s3_object_body = s3_response.get("Body")

        # Read the data in bytes format
        content = s3_object_body.read()

        try:
            # Parse JSON content to Python Dictionary
            json_dict = json.loads(content)

            # Print the file contents as a string
            return json_dict

        except json.decoder.JSONDecodeError as e:
            # JSON is not properly formatted
            logging.info("JSON file is not properly formatted")
            logging.info(e)

    except s3_resource.meta.client.exceptions.NoSuchBucket as e:
        # S3 Bucket does not exist
        logging.info("NO SUCH BUCKET")
        logging.info(e)

    except s3_resource.meta.client.exceptions.NoSuchKey as e:
        # Object does not exist in the S3 Bucket
        logging.info("NO SUCH KEY")
        logging.info(e)

def save_dict(dict: dict,
              bucket_name: str,
              path_file: str,
              filename: str=None) -> None:
    """Saves a dictionary into a json file

    Args:
        dict (dict): dictionary to save
        bucket_name (str): name of the bucket (not ending with '/')
        folder_path (str): path to folder containg file
        data_file (str): file name
    Returns:
        None
    """
    metadata_encoded = json.dumps(dict)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).put_object(
        Key=os.path.join(path_file, filename), Body=metadata_encoded
    )

def save_text(text: str,
              bucket_name: str,
              path_file: str,
              filename: str=None) -> None:
    """Saves a dictionary into a json file

    Args:
        text (str): text to save
        bucket_name (str): name of the bucket (not ending with '/')
        folder_path (str): path to folder containg file
        data_file (str): file name
    Returns:
        None
    """
    if filename:
        path_file = os.path.join(path_file, filename)
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).put_object(
        Key=path_file, Body=text
    )

def save_figure_s3(
    figure_object: plotly.graph_objs.Figure, 
    bucket_name: str,
    path_file: str,
    filename: str,
):
    """Save figure (matplotlib or plotly) to S3 bucket in png format.

    Args:
        figure_object : figure object
        bucket_name (str): name of bucket
        folder_path (str): path to file
        file_name (str): file name ending with ".png"
    """
    s3 = boto3.resource("s3")
    filepath = os.path.join(path_file, filename)

    if type(figure_object) == plotly.graph_objs._figure.Figure:
        # initialiaze io to_bytes converter
        imdata = BytesIO()
        # define quality of saved array
        figure_object.write_image(imdata, format="PNG")

    elif type(figure_object) == matplotlib.figure.Figure:
        # Initialize figure object
        canvas = FigureCanvasAgg(figure_object)  # renders figure onto canvas
        imdata = (
            BytesIO()
        )  # prepares in-memory binary stream buffer (think of this as a txt file but purely in memory)
        canvas.print_png(
            imdata
        )  # writes canvas object as a png file to the buffer. You can also use print_jpg, alternatively

    # this makes a new object in the bucket and puts the file in the bucket
    # ContentType parameter makes sure resulting object is of a 'image/png' type and not a downloadable 'binary/octet-stream'
    s3.Object(bucket_name, filepath).put(
        Body=imdata.getvalue(), ContentType="image/png"
    )

def load_model(bucket_name: str, folder_path: str, model_file: str):
    """Loads model in a pickle format

    Args:
        bucket_name (str): name of AWS bucket
        folder_path (str): path to file
        model_file (str): file name ending  with ".pkl"

    Returns:
        model 
    """
    logging.info(
        "Model will be loaded in {}".format(
            os.path.join("s3://", bucket_name, folder_path, model_file)
        )
    )
    s3 = boto3.resource("s3")
    model = pickle.loads(
        s3.Bucket(bucket)
        .Object(os.path.join(folder_path, model_file))
        .get()["Body"]
        .read()
    )
    return model

def save_model(model,
               bucket_name: str,
               folder_path: str, 
               file_name):
    """Save mode (that could be pickled with pickle.dumps) to S3 bucket in pickle format

    Args:
        model : model that could be pickled
        bucket_name (str): name of bucket
        folder_path (str): path to file
        file_name (str): file name ending  with ".pkl"
    """
    logging.info(
        "Model will be saved in {}".format(
            os.path.join("s3://", bucket_name, folder_path, file_name)
        )
    )
    s3_resource = boto3.resource("s3")
    pickle_byte_obj = pickle.dumps(model)
    s3_resource.Object(bucket, os.path.join(folder_path, file_name)).put(
        Body=pickle_byte_obj
    )

# Referentials # 

def load_variables_referential():
    return pd.read_excel(
        io=os.path.join(conf.PATH_CONF, conf.FILENAME_FEATURE_REFERENTIAL),
        sheet_name=conf.REFERENTIAL_INFORMATION_SHEETNAME,
    )


def load_variables_referential_dict(df_ref_variables: Optional[pd.DataFrame] = None):
    if df_ref_variables is None:
        df_ref_variables = load_variables_referential()

    # Convert variable referential to dict
    ref_variables_dict = df_ref_variables.set_index(
        conf.REFERENTIAL_VAR_NAME
    ).to_dict("index")

    return ref_variables_dict


def load_variable_usage_referential():
    return pd.read_excel(
        io=os.path.join(conf.PATH_CONF, conf.FILENAME_FEATURE_REFERENTIAL),
        sheet_name=conf.REFERENTIAL_USAGE_SHEETNAME,
    )


def load_variable_usage_referential_dict(
    df_ref_var_usage: Optional[pd.DataFrame] = None,
):
    if df_ref_var_usage is None:
        df_ref_var_usage = load_variable_usage_referential()

    # Convert variable referential to dict
    ref_var_usage_dict = df_ref_var_usage.set_index(
        conf.REFERENTIAL_VAR_NAME
    ).to_dict("index")

    return ref_var_usage_dict

