import torch

def normalize_tensor(input_tensor):
    '''Normalizes a tensor of any dimensions. Goal is to have inputs on the range 0-1, NOT a norm of 1.'''
    
    # Compute min and max values across all dimensions
    min_values = torch.min(input_tensor)
    max_values = torch.max(input_tensor)

    return (input_tensor - min_values) / (max_values - min_values)