import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdmetrics.reports.single_table import QualityReport
from src.utils.utils_df import add_primary_key

def get_metadata_from_df(df: pd.DataFrame) -> dict:
    """Create a metadata from dataframe"""
    # create single table metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    return metadata

def check_metadata(metadata: SingleTableMetadata(),
                   primary_key: str) -> bool:
    """Check if metadata is valid"""
   
    # check whether all columns have a known type 
    list_col_unknown = metadata.get_column_names(sdtype='unknown')
    assert  list_col_unknown == [], f"{list_col_unknown} columns have unknown type"
    
    dict_metadata = metadata.to_dict()
    # check that primary key is valid
    assert dict_metadata.get('primary_key') == primary_key, f"Primary key is not valid"
    
    return None

def get_metadata_from_dict(dict_metadata: pd.DataFrame) -> dict:
    """Create a SDV metadata from a metadata in python dictionary format"""
    sdv_metadata = SingleTableMetadata.load_from_dict(dict_metadata)
    return sdv_metadata

def get_sdv_report(
    df_real: pd.DataFrame,
    df_synth: pd.DataFrame,
    sdv_metadata: SingleTableMetadata(),
) -> QualityReport:
    """Retrieves a sdv Quality report

    Args:
        df_real (pd.DataFrame): preprocessed original cohort data, 1 row/ patient & 1 col/ variable
        df_synth (pd.DataFrame): simulated cohort data, 1 row/ patient & 1 col/ variable
        sdv_metadata (SingleTableMetadata): sdv metadata

    Returns:
        QualityReport: sdv quality report
    """
    sdv_report = evaluate_quality(df_real, df_synth, sdv_metadata)
    return sdv_report

def get_mapping_type():
    mapping = {'categorical': int,
               'boolean': int,
               'numerical': float,
               'id': 'id', # TODO change this
               }
    return mapping

def infer_type_from_metadata(df: pd.DataFrame,
                             metadata: SingleTableMetadata()) -> pd.DataFrame:
    
    mapping_type = get_mapping_type()
    
    for k, v in metadata.to_dict().get("columns").items():
        # get python mapping
        type = v.get("sdtype")
        python_type = mapping_type.get(type)
        assert python_type, f"{type} not found in mapping_type"
        if python_type != 'id':
            # transform to correct type
            df[k] = df[k].astype(python_type)
    return df

def custom_validate_data(df: pd.DataFrame,
                        metadata: SingleTableMetadata()) -> pd.DataFrame:
    """Validate data with metadata. Transform type of columns if incorrect"""
    # create fictious primary key column if not present
    # this is necessary for using metrics nedding metadata dict that necessary has a primary key
    primary_key = metadata.to_dict().get('primary_key')
    df = add_primary_key(df=df,
                         primary_key=primary_key)
    
    # check that all columns from metadata are in data
    list_cols = metadata.get_column_names()
    all_cols_in_list_bool = all(col in df.columns for col in list_cols)
    assert all_cols_in_list_bool, f"Columns not found in metadata"
    
    # validate data with metadata
    # NB: this does not validate the types of the columns which is still problematic
    metadata.validate_data(data=df)
    
    # infers type from metadata sdype mapping
    df = infer_type_from_metadata(df=df,
                                  metadata=metadata)
    
    # for sdv plots sort columns in alphabetical order 
    df = df.sort_index(axis=1)
    
    return df
    


