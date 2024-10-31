""" Functions to handle feature referential file"""

from typing import Union
import config

from src import loading

## Variable information referential


def filter_ref(ref: dict, key: str, value: Union[str, int, list]) -> dict:
    """Filters referential on variables respecting a rule"""
    ref_filt = {}

    for k, v in ref.items():
        if type(value) == list:
            condition = v.get(key) in value
        else:
            condition = v.get(key) == value
        if condition:
            ref_filt[k] = v
    return ref_filt


def get_list_vars(ref: dict) -> list:
    """Retrieves variables of referential as list of variables"""
    return list(ref.keys())


def get_dynamic_vars(ref: dict) -> list:
    """Retrieves list of dynamic variables from referential"""
    filt_ref = filter_ref(ref=ref, key=config.REFERENTIAL_VAR_NATURE, value="dynamic")
    list_vars_dyn = get_list_vars(ref=filt_ref)
    return list_vars_dyn


def get_time_related_vars(ref: dict) -> list:
    """Retrieves list of dynamic variables from referential"""
    filt_ref = filter_ref(
        ref=ref, key=config.REFERENTIAL_VAR_NATURE, value="time_related"
    )
    list_vars_dyn = get_list_vars(ref=filt_ref)
    return list_vars_dyn


def get_static_vars(ref: dict) -> list:
    """Retrieves list of dynamic variables from referential"""
    filt_ref = filter_ref(ref=ref, key=config.REFERENTIAL_VAR_NATURE, value="static")
    list_vars_static = get_list_vars(ref=filt_ref)
    return list_vars_static


def get_disc_vars(ref: dict) -> list:
    """Retrieves list of dynamic variables from referential"""
    filt_ref = filter_ref(ref=ref, key=config.REFERENTIAL_VAR_TYPE, value="int")
    list_vars_discr = get_list_vars(ref=filt_ref)
    return list_vars_discr


def get_cont_vars(ref: dict) -> list:
    """Retrieves list of dynamic variables from referential"""
    filt_ref = filter_ref(ref=ref, key=config.REFERENTIAL_VAR_TYPE, value="float")
    list_vars_cont = get_list_vars(ref=filt_ref)
    return list_vars_cont


def get_var_mapping(ref: dict) -> dict:
    """Retrieves mapping between original variable names and new variable names of static variables"""
    mapping = {}
    for k, v in ref.items():
        var_name_raw = v.get(config.REFERENTIAL_VAR_NAME_PPMI)
        var_name = k
        if isinstance(var_name_raw, str):
            mapping[var_name_raw] = var_name
    return mapping


def get_var_dyn_mapping(ref: dict, nb_events: int) -> dict:
    """Retrieves mapping between original variable names and new variable names of dynamic variables"""
    ref = filter_ref(ref=ref, key=config.REFERENTIAL_VAR_NATURE, value="dynamic")
    mapping = {}
    for k, v in ref.items():
        var_name_new = v.get(config.REFERENTIAL_VAR_NAME)
        for i in range(nb_events):
            mapping[f"{k}_{i}"] = f"{var_name_new} (visit {i + 1})"
    return mapping


def get_var_all_mapping(ref: dict, nb_events: int) -> dict:
    """Retrieves mapping between original variable names and new variable names of static & dynamic variables"""
    var_mapping = get_var_mapping(ref=ref)
    var_dyn_mapping = get_var_dyn_mapping(ref=ref, nb_events=nb_events)
    var_mapping.update(var_dyn_mapping)
    return var_mapping


def map_list(L: list, mapping_vars: dict) -> dict:
    return [mapping_vars.get(x) for x in L]


## Variable usage referential


def get_variables_to_keep(ref: dict):
    ref = filter_ref(ref=ref, key=config.REFERENTIAL_USE_MODELLING, value=1)
    return get_list_vars(ref)


def get_var_list_by_nature(ref_var: dict, ref_var_usage: dict):
    """Get list of static, time-related and dynamic variables from referential"""
    # Get variables of interest from variable usage referential
    list_variables_to_keep = get_variables_to_keep(ref=ref_var_usage)

    # Get static & dynamic variables from variable information referential
    list_static_vars = get_static_vars(ref=ref_var)
    list_time_related_vars = get_time_related_vars(ref=ref_var)
    list_dynamic_vars = get_dynamic_vars(ref=ref_var)

    # Get final variable list
    list_static_vars = [
        var for var in list_static_vars if var in list_variables_to_keep
    ]
    list_time_related_vars = [
        var for var in list_time_related_vars if var in list_variables_to_keep
    ]
    list_dynamic_vars = [
        var for var in list_dynamic_vars if var in list_variables_to_keep
    ]

    return list_static_vars, list_time_related_vars, list_dynamic_vars

def get_ref_variables_to_keep(ref_var: dict=None, ref_var_usage: dict=None):

    if ref_var is None:
        ## Load variable information
        ref_var = loading.load_variables_referential_dict()
    if ref_var_usage is None:
        ## Load variable usage information
        ref_var_usage = loading.load_variable_usage_referential_dict()
    
    ## Get variables of interest from variable usage referential
    list_variables_to_keep = get_variables_to_keep(ref=ref_var_usage)
    ## Get referential of variable of interest
    ref_variables_kept_dict = {k: v for k, v in ref_var.items() if k in list_variables_to_keep}

    return ref_variables_kept_dict