"""
© 2025, Stefan Webb. Some Rights Reserved.

Except where otherwise noted, this work is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International
https://creativecommons.org/licenses/by-sa/4.0/deed.en

"""

def count_parameters(model):
    total_elements = 0
    state_dict = model.state_dict()
    for param_tensor in state_dict:
        total_elements += state_dict[param_tensor].numel()
    return total_elements