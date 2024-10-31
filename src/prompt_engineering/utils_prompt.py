import numpy as np
from src.utils import utils_referential
from src.utils.utils_df import shuffle_dict


def parse_prompt(prompt_dict: dict,
                 prompt_example: dict,
                 var_desc_prompt_dict: dict,
                 ref_key: str,
                 shuffle: bool=False):
    """Parse prompt from prompt template dictionary

    Args:
        prompt_dict (dict): template dictionary prompt
        prompt_example (dict): row examples of output
        var_desc_prompt_dict (dict): dictionary containing data specifications dictionary
        ref_key (str): key to use in reference dictionary
        shuffle (bool, optional): whether to shuffle or not the variables. Defaults to False.

    Returns:
        prompt in string format
    """
    # get prompt or tempalte prompt from template dictionary
    prompt = prompt_dict["prompt"]
    if not prompt_dict["is_template"] or prompt_dict["template_items"] is None:
        output_prompt = prompt
        
    # create prompt from template items and template prompt
    else:
        
        # prompt items dictionary
        prompt_items_dict = dict()
        for prompt_item in prompt_dict["template_items"]:
            prompt_items_dict[prompt_item] = parse_prompt_item(item=prompt_item,
                                                               prompt_example=prompt_example,
                                                               var_desc_prompt_dict=var_desc_prompt_dict,
                                                               ref_key=ref_key,
                                                                shuffle=shuffle)
        output_prompt = prompt.format(**prompt_items_dict)
    
    return output_prompt


def parse_prompt_item(item: str,
                      prompt_example: dict,
                      var_desc_prompt_dict: dict,
                      ref_key: str,
                      shuffle: bool=False):
    """Parse prompt item from prompt template dictionary
    
    Args:
        item (str): item to parse
        prompt_example (dict): row examples of output
        var_desc_prompt_dict (dict): dictionary containing data specifications dictionary
        ref_key (str): key to use in reference dictionary
        shuffle (bool, optional): whether to shuffle or not the variables. Defaults to False.
        
    Returns:
        prompt item in string format
    """
    
    if item == "row_example":
         return prompt_example
    
    elif item == "variables_description":
        
        # Get variable description referential
        ref_variables = utils_referential.get_ref_variables_to_keep()
        
        if shuffle:
            ref_variables = shuffle_dict(d=ref_variables)
        return get_prompt_desc_all_variables(
            ref=ref_variables,
            var_desc_prompt_template=var_desc_prompt_dict["template"],
            var_desc_prompt_template_mapping=var_desc_prompt_dict["mapping"],
            ref_key=ref_key
            )

    else:
        raise ValueError(f"Request item {item} not implemented")
    
def get_prompt_desc_all_variables(ref: list,
                                 var_desc_prompt_template: str,
                                 var_desc_prompt_template_mapping: dict,
                                 ref_key: str):
    """
    Get prompt description for all variables in referential 
    
    Args:
        ref (list): referential list of variables
        var_desc_prompt_template (str): template for variable description
        var_desc_prompt_template_mapping (dict): mapping of template variables
        ref_key (str): key to use in reference dictionary

    Returns:
        variables_description in string format
    """
    variable_description_list = list()
    for var_name, var_dict in ref.items():
        var_description = get_prompt_desc_var(var_dict=var_dict, 
                                            var_name=var_name, 
                                            var_desc_prompt_template=var_desc_prompt_template, 
                                            var_desc_prompt_template_mapping=var_desc_prompt_template_mapping, 
                                            ref_key=ref_key)
        variable_description_list.append(var_description)
    variables_description = '\n'.join(variable_description_list)
    return variables_description

def get_prompt_desc_var(var_name: str, 
                        var_dict: dict, 
                        var_desc_prompt_template: str,
                        var_desc_prompt_template_mapping: dict,
                        ref_key: str) -> str:
    description_dict = dict()
    # Get description arguments from template
    for template_name, item in var_desc_prompt_template_mapping.items():
        if item == ref_key:
            description_dict[template_name] = var_name
        elif item in var_dict.keys():
            if type(var_dict[item]) == float and np.isnan(float(var_dict[item])):
                description_dict[template_name] = ""
            else:
                description_dict[template_name] = var_dict[item]
        else:
            description_dict[template_name] = ""

    # Compute variable prompt description from template
    return var_desc_prompt_template.format(**description_dict)




