"""
    Assume input tensor is of the form:
    tensor = [outlook,temp,humidity,windy,play]
    here play is the target variable (class)
    remaining four are explanatory variables

"""
import torch
import math
from statistics import mean

"""Calculate the entropy of the entire dataset"""


# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor: torch.Tensor):
    # TODO
    p_pos = (list(tensor[:, tensor.size(1) - 1]).count(1)) / tensor.size(0)

    p_neg = list(tensor[:, tensor.size(1) - 1]).count(0) / tensor.size(0)

    entropy = (-1 * p_pos * math.log(p_pos, 2)) - (p_neg * math.log(p_neg, 2))

    return entropy
    pass


"""Return avg_info of the attribute provided as parameter"""


# input:tensor,attribute number
# output:int/float
def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    # TODO
    column = tensor[:, attribute].tolist()
    
    values = set(column)
    
    summation = 0
    for i in values:
        value_count = column.count(i)
        
        pos = 0
        neg = 0
        for j in range(len(column)):
            if column[j] == i and list(tensor[:, tensor.size(1) - 1])[j] == 1:
                pos += 1
            elif column[j] == i and list(tensor[:, tensor.size(1) - 1])[j] == 0:
                neg += 1
        if neg == 0 or pos ==0:
            summation += 0
        else:
            summation += (value_count / tensor.size(0)) * (
                (-1 * pos / value_count * math.log((pos / value_count), 2))
                - (neg / value_count * math.log((neg / value_count), 2))
            )
    return summation
    pass


"""Return Information Gain of the attribute provided as parameter"""


# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor: torch.Tensor, attribute: int):
    # TODO

    info_gain = get_entropy_of_dataset(tensor) - get_avg_info_of_attribute(
        tensor, attribute
    )
    return info_gain
    pass


# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor: torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    # TODO
    dictionary = {}
    
    for key in range(tensor.size(1)-1):
        dictionary[key] = get_information_gain(tensor, key)
    
    max_key = max(zip(dictionary.values(), dictionary.keys()))[1]
    
    result = (dictionary, max_key)
    return result
    pass
